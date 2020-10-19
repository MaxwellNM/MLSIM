# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
from parameters import *
from utils import *
import time
import os
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score, auc,precision_recall_curve

class RBM(object):
    def __init__(self, input=None, n_visible=2, n_hidden=3, \
        W=None, hbias=None, vbias=None, randomizer=None,batch_size=50,learning_rate=1e-3,epochs=100,verbose=True):
        
        self.n_visible = n_visible  # num of units in visible (input) layer
        self.n_hidden = n_hidden    # num of units in hidden layer
        self.batch_size =batch_size
        self.learning_rate=learning_rate
        self.n_epochs = epochs
        self.verbose = verbose

        # if randomizer is None:
        #     randomizer = np.random.RandomState(1234)


        if W is None:
            a = 1. / n_visible
            initial_W = np.array(randomizer.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(n_visible, n_hidden)))

            W = initial_W

        if hbias is None:
            hbias = np.zeros(n_hidden)  # initialize h bias 0

        if vbias is None:
            vbias = np.zeros(n_visible)  # initialize v bias 0


        np = randomizer
        self.input = input
        self.W = W
        self.hbias = hbias
        self.vbias = vbias

    def set_weigths(self,weigth_dic):
        self.W = weigth_dic["W"]
        self.hbias = weigth_dic["h"]
        self.vbias =  weigth_dic["v"]      

    def _stochastic_gradient_descent(self, _data,rbm_index,verbose=False):
        """
        Performs stochastic gradient descend optimization algorithm.
        :param _data: array-like, shape = (n_samples, n_features)
        :return:
        """
        accum_delta_W = np.zeros(self.W.shape)
        accum_delta_b = np.zeros(self.hbias.shape)
        accum_delta_c = np.zeros(self.vbias.shape)
        train_history = []
        for iteration in range(1, self.n_epochs + 1):
            idx = np.random.permutation(len(_data))
            data = _data[idx]
            for batch in batch_generator(self.batch_size, data):
                accum_delta_W[:] = .0
                accum_delta_b[:] = .0
                accum_delta_c[:] = .0
                for sample in batch:
                    sample_expend = np.expand_dims(sample,0)
                    #print ("sample: ",np.shape(sample)," val: ",sample)
                    delta_W, delta_b, delta_c = self.contrastive_divergence(self, k=HYPERPARAMS.constrative_k,input=sample)
                    accum_delta_W += delta_W
                    accum_delta_b += delta_b
                    accum_delta_c += delta_c
                self.W += self.learning_rate * (accum_delta_W / self.batch_size) 
                self.hbias += self.learning_rate * (accum_delta_b / self.batch_size)
                self.vbias += self.learning_rate * (accum_delta_c / self.batch_size)
            if TRAINING.save_history:
                #compute weights amplitude over training
                error = self.get_reconstruction_cross_entropy(data) #_compute_reconstruction_error(data)
                weights_magnitude = frobinus_norm(self.W)
                b_magnitude = frobinus_norm(self.hbias)
                v_magnitude = frobinus_norm(self.vbias)
                train_history.append({ 'epoch':iteration,'RBM':rbm_index+1,  'cost':error})
                if verbose:
                    print(">> Epoch %d finished \tRBM %d Reconstruction error %f \tWeigths %f hbias %f vbias %f  " % (iteration,rbm_index+1, error,weights_magnitude,b_magnitude,v_magnitude))
        return self.W,self.hbias,self.vbias



    """Machine Learning: An Algorithmic Perspective, Second Edition https://www.goodreads.com/book/show/20607838-machine-learning
       by Stephen Marsland pp.372-373"""
    def contrastive_divergence(self, lr=0.01, k=1, input=None):
        W = np.zeros(self.W.shape)
        hbias = np.zeros(self.hbias.shape)
        vbias = np.zeros(self.vbias.shape)     
        if input is not None:
            self.input = input

        ''' Constrastive divergence - k '''
        ph_mean, ph_sample = self.sample_h_given_v(input)
        chain_start = ph_sample
        #Gibbs Sampling
        for step in range(k):
            if step == 0:
                nv_means, nv_samples,\
                nh_means, nh_samples = self.gibbs_hvh(chain_start)
            else:
                nv_means, nv_samples,\
                nh_means, nh_samples = self.gibbs_hvh(nh_samples)

        #chain_end = nv_samples
        extend_inp = np.expand_dims(input,0)

        val1 = np.dot(ph_mean.T,extend_inp)
        val2 = np.dot(nh_means.T,nv_samples)
        W += self.learning_rate * (val1-val2)
        vbias += self.learning_rate * np.mean(extend_inp- nv_samples, axis=0)
        hbias += self.learning_rate * np.mean(ph_mean - nh_means, axis=0)
        #cost = self.get_reconstruction_cross_entropy()
        return W, hbias, vbias  #cost

    def sample_h_given_v(self, v0_sample):
        h1_mean = self.propup(v0_sample)
        h1_sample = np.random.binomial(size=h1_mean.shape,   # discret: binomiale
                                       n=1,
                                       p=h1_mean)
        return [h1_mean, h1_sample] #h1_mean est probabilité d'apparition d'un 0 ou 1 pour la couche caché.
                                    #h1_sample est la séquence binaire (0 ou 1) génerée par la loi de Bernouilli (binomiale) suivant la probabilité h1_mean


    def sample_v_given_h(self, h0_sample):
        v1_mean = self.propdown(h0_sample)
        v1_sample = np.random.binomial(size=v1_mean.shape,   # discret: binomiale
                                            n=1,
                                            p=v1_mean)
        
        return [v1_mean, v1_sample]

    def propup(self, v):
        pre_sigmoid_activation = np.dot(v, self.W.T) + self.hbias
        return sigmoid(pre_sigmoid_activation)

    def propdown(self, h):
        pre_sigmoid_activation = np.dot(h, self.W) + self.vbias
        return sigmoid(pre_sigmoid_activation)


    def gibbs_hvh(self, h0_sample):
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        h1_mean, h1_sample = self.sample_h_given_v(v1_sample)

        return [v1_mean, v1_sample,
                h1_mean, h1_sample]
    

    def get_reconstruction_cross_entropy2(self,sample):
        pre_sigmoid_activation_h = np.dot(sample, self.W.T) + self.hbias
        sigmoid_activation_h = sigmoid(pre_sigmoid_activation_h)
        
        pre_sigmoid_activation_v = np.dot(sigmoid_activation_h, self.W) + self.vbias
        sigmoid_activation_v = sigmoid(pre_sigmoid_activation_v)
        cross_entropy =  - np.mean(
            np.sum(sample - np.log(sigmoid_activation_v)**2 ,
                      axis=1))
        
        return cross_entropy

    def get_reconstruction_cross_entropy(self,sample):
        pre_sigmoid_activation_h = np.dot(sample, self.W.T) + self.hbias
        sigmoid_activation_h = sigmoid(pre_sigmoid_activation_h)
        
        pre_sigmoid_activation_v = np.dot(sigmoid_activation_h, self.W) + self.vbias
        sigmoid_activation_v = sigmoid(pre_sigmoid_activation_v)
        #print('Activation ',sigmoid_activation_v)
        #print('Input ',self.input)
        #cost = x*log(x_tilde) +(1-x)*log(1-x_tilde)
        cross_entropy =  - np.mean(
            np.sum(sample * np.log(sigmoid_activation_v) +
            (1 - sample) * np.log(1 - sigmoid_activation_v),
                      axis=1))
        
        return cross_entropy

    def reconstruct(self, v):
        h = sigmoid(np.dot(v, self.W) + self.hbias)
        reconstructed_v = sigmoid(np.dot(h, self.W.T) + self.vbias)
        return reconstructed_v

class MDBN(object):
    def __init__(self, input_data=None, label=None,  input_size=2, hidden_layer_sizes=[3, 3],batch_size=HYPERPARAMS.batch_size, learning_rate=0.1,epochs=50,C=1):
        
        self.input = input_data.astype(np.float32)
        self.y = label.astype(np.float32)
        self.rbm_layers = []
        self.n_hidden_layers = len(hidden_layer_sizes)  # nombre de couches cachées du DBN
        self.batch_size = batch_size
        self.layer_svm = None
        self.model_finetune = None
        self.nepochs = epochs
        self.learning_rate = learning_rate
        self.C = C
        self.weights = dict()
        W,hbias,vbias = self.init_weights(input_size,hidden_layer_sizes[0])
        # construct rbm_layer
        rbm_layer = RBM(input=self.input,
                        n_visible=input_size,
                        n_hidden=hidden_layer_sizes[0],
                        W=W,  
                        hbias=hbias,
                        vbias=vbias,
                        batch_size=self.batch_size,
                        learning_rate=self.learning_rate ,
                        epochs=self.nepochs)
        self.weights[0] = {"W":W,"h":hbias,"v":vbias}
        self.rbm_layers.append(rbm_layer)
        sample_input =self.input
        i = 1
        for rbm_index in range(self.n_hidden_layers-1):
            _,sample_input =   self.rbm_layers[rbm_index].sample_h_given_v(sample_input)    
            n_hid =  hidden_layer_sizes[rbm_index+1]
            n_vis =  hidden_layer_sizes[rbm_index]
            W,hbias,vbias = self.init_weights(n_vis,n_hid)
            self.weights[i] = {"W":W,"h":hbias,"v":vbias}
            rbm_i = RBM(input=sample_input,
                        n_visible = n_vis,
                        n_hidden = n_hid,
                        W=W,  
                        hbias=hbias,
                        vbias=vbias,
                        batch_size=self.batch_size,
                        learning_rate=self.learning_rate ,
                        epochs=self.nepochs)
            i+=1
            self.rbm_layers.append(rbm_i)

    def load_weights(self, X_train):
        i=0
        for rbm_index in range(self.n_hidden_layers):    
            self.rbm_layers[rbm_index].set_weigths(self.weights[i] )
            i+=1
    
    def save_weigths(self,path_folder):
        np.save(path_folder+"/MLSIM_weigths.npy",self.weights)


    def train(self, X_train,verbose=False):
        #training
        i=0
        for rbm_index in range(self.n_hidden_layers):
            rbm =   self.rbm_layers[rbm_index]

            W,hbias,vbias = rbm._stochastic_gradient_descent(X_train,rbm_index,verbose)
            self.weights[i] = {"W":W,"h":hbias,"v":vbias}
            _,X_trainp = rbm.sample_h_given_v(X_train) #kind prediction last layer 
            X_train = X_trainp
            #save weights
            self.weights[i] = {"W":W,"h":hbias,"v":vbias}
            i+=1


    def init_weights(self,n_visible,number_neuron_hiden):
        W = None
        hbias = None
        vbias = None
        a = 1./np.sqrt(n_visible)
        b = 1./np.sqrt(number_neuron_hiden)

        if W is None:
            a = 1. / n_visible
            initial_W = np.array(np.random.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(number_neuron_hiden,n_visible)))

            W = initial_W
            #print("Initialize weigths",W)

        if hbias is None:
            hbias =np.array(np.random.uniform(  # initialize h bias
                low=-b,
                high=b,
                size=(1, number_neuron_hiden)))
        if vbias is None:
            vbias = np.array(np.random.uniform(  # initialize v bias
                low=-a,
                high=a,
                size=(1, n_visible)))
        return W, hbias,vbias

    #No training weigths
    def predict(self, X_predict):

        h_list = []
        synthetic_V_list = []
        for rbm in self.rbm_layers:
            #print ("shape Xpredict ",np.shape(X_predict))
            hvar_mean, hvar_sample = rbm.sample_h_given_v(X_predict)
            #print ("shape hvar ",np.shape(hvar_sample))
            vvar_mean,vvar_sample = rbm.sample_v_given_h(hvar_sample)
            #print ("shape vvar ",np.shape(vvar_sample))
            X_predict = hvar_sample
            h_list.append(hvar_mean)
            synthetic_V_list.append(vvar_mean)
        #print ("shape last layer: ", h_list[0].shape,h_list[1].shape)

        return synthetic_V_list, h_list #h_list[-1] est la dernière couche du Réseau de RBM à utiliser pour finetuner

    def finetune(self,X_transfer, y_transfer,msg,rng=1245,classifier="SVM",weight=False, bytraining=True):
        #synthetic_V_list, h_list = self.predict(X_transfer)
        X_finetuned = self.get_last_layer_feature(X_transfer)#h_list[-1] #new_data_entry
        if bytraining:
            if classifier=="SVM":
                print('Training with SVM')
                #train the last layer with hybrid model
                #train last layer with SVM
                if weight:
                    cw = {}
                    for l in set(y_transfer):
                        cw[l] = np.sum(y_transfer == l)
                    self.model_finetune = SVC(C=1,probability=True,cache_size=200,max_iter=1000,class_weight=cw,random_state=rng,verbose=True) #DecisionTreeClassifier(criterion="entropy", max_depth=1) #LogisticRegression ()
                else:
                    self.model_finetune = SVC(C=self.C,probability=True,cache_size=200,random_state=rng,verbose=True) 
                print('Finetuning with SVM')
                strt2=time.time()
                self.model_finetune.fit(X_finetuned,y_transfer)
                print ("Time fitting the data: ",time.time()-strt2)
            elif classifier=="MLP":
                #train the last layer with hybrid model
                #train last layer with MLP
                self.model_finetune = MLPClassifier(max_iter=500,solver="adam",random_state=12345,learning_rate_init=0.01,verbose=True) 
                print('Finetuning with MLP')
                strt2=time.time()
                self.model_finetune.fit(X_finetuned,y_transfer)
                print ("Time fitting the data: ",time.time()-strt2)                
        predictions = self.model_finetune.predict(X_finetuned)
        return predictions
        
    
    #Forwadings
    def get_last_layer_feature(self,data):
        synthetic_V_list, h_list = self.predict(data)
        X_finetuned = h_list[-1]
        return X_finetuned


    def evaluate(self, X_test,y_test,msg,verbose=False):
        predictions = self.finetune(X_test,y_test,msg, bytraining=False)
        precision, recall, thresholds = precision_recall_curve(y_test, predictions)
        aucc = auc(recall, precision)#accuracy_score(y_test,predictions)
        if verbose:
            #print (msg," Model accuracy :",accuracy_score(y_test,predictions))
            print (msg," Confusion matrix :\n",confusion_matrix(y_test,predictions))
        return aucc
    
    def getweights(self):
        return self.weights
