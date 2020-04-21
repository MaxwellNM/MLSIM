# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
#from .HiddenLayer import HiddenLayer
#from .LogisticRegression import LogisticRegression
from parameters import *
from utils import *
import time
import os
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score

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

    def _stochastic_gradient_descent(self, _data,rbm_index):
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
                error = self.get_reconstruction_cross_entropy2(data) #_compute_reconstruction_error(data)
                train_history.append({ 'epoch':iteration,'RBM':rbm_index+1,  'cost':error})
                print(">> Epoch %d finished \tRBM %d Reconstruction error %f" % (iteration,rbm_index+1, error))
        return train_history



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
        #extend_nv_samples = np.expand_dims(nv_samples,0)
        
        # print("Dimension W :",np.shape(W))
        # print("Dimension synthetic v :",np.shape(nv_samples))
        # print("Dimension predict h :",np.shape(nh_samples))
        # print("Dimension synthetic v mean :",np.shape(nv_means))
        # print("Dimension predict h mean :",np.shape(nh_means))
        #print("Dimension input: ",np.shape(extend_inp))
        # print("lr = ",self.learning_rate)
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
            np.sum(sample * np.log(sigmoid_activation_v) +
            (1 - sample) * np.log(1 - sigmoid_activation_v),
                      axis=1))
        
        return cross_entropy

    def get_reconstruction_cross_entropy(self):
        pre_sigmoid_activation_h = np.dot(self.input, self.W) + self.hbias
        sigmoid_activation_h = sigmoid(pre_sigmoid_activation_h)
        
        pre_sigmoid_activation_v = np.dot(sigmoid_activation_h, self.W.T) + self.vbias
        sigmoid_activation_v = sigmoid(pre_sigmoid_activation_v)
        #print('Activation ',sigmoid_activation_v)
        #print('Input ',self.input)
        #cost = x*log(x_tilde) +(1-x)*log(1-x_tilde)
        cross_entropy =  - np.mean(
            np.sum(self.input * np.log(sigmoid_activation_v) +
            (1 - self.input) * np.log(1 - sigmoid_activation_v),
                      axis=1))
        
        return cross_entropy

    def reconstruct(self, v):
        h = sigmoid(np.dot(v, self.W) + self.hbias)
        reconstructed_v = sigmoid(np.dot(h, self.W.T) + self.vbias)
        return reconstructed_v

class MDBN(object):
    def __init__(self, input_data=None, label=None,  input_size=2, hidden_layer_sizes=[3, 3], output_size=1,batch_size=HYPERPARAMS.batch_size, learning_rate=0.1,epochs=50,C=1):
        
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
        self.rbm_layers.append(rbm_layer)
        sample_input =self.input
        for rbm_index in range(self.n_hidden_layers-1):
            _,sample_input =   self.rbm_layers[rbm_index].sample_h_given_v(sample_input)    
            n_hid =  hidden_layer_sizes[rbm_index+1]
            n_vis =  hidden_layer_sizes[rbm_index]
            W,hbias,vbias = self.init_weights(n_vis,n_hid)
            rbm_i = RBM(input=sample_input,
                        n_visible = n_vis,
                        n_hidden = n_hid,
                        W=W,  
                        hbias=hbias,
                        vbias=vbias,
                        batch_size=self.batch_size,
                        learning_rate=self.learning_rate ,
                        epochs=self.nepochs)
            self.rbm_layers.append(rbm_i)

  

    def train(self, X_train):
        #training
        #print("start the training: ",np.shape(X_train))
        #print("Number of layers ",len(self.rbm_layers))
        for rbm_index in range(self.n_hidden_layers):
            rbm =   self.rbm_layers[rbm_index]
            #print ("Index rbm ",rbm_index)
            #print("regenerate X_train: ",np.shape(X_train))
            rbm._stochastic_gradient_descent(X_train,rbm_index)
            _,X_trainp = rbm.sample_h_given_v(X_train) #kind prediction last layer 
            X_train = X_trainp
            #save weights


    def init_weights(self,n_visible,number_neuron_hiden):
        W = None
        hbias = None
        vbias = None
        a = 1./np.sqrt(n_visible)
        b = 1./np.sqrt(number_neuron_hiden)
        #v = a
        if W is None:
            a = 1. / n_visible
            initial_W = np.array(np.random.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(number_neuron_hiden,n_visible)))

            W = initial_W
            print("Initialize weigths",W)

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
            h_list.append(hvar_sample)
            synthetic_V_list.append(vvar_sample)
        #print ("shape last layer: ", h_list[0].shape,h_list[1].shape)

        return synthetic_V_list, h_list #h_list[-1] est la dernière couche du Réseau de RBM à utiliser pour finetuner

    def finetune(self,X_transfer, y_transfer,msg, bytraining=True):
        #synthetic_V_list, h_list = self.predict(X_transfer)
        X_finetuned = self.get_last_layer_feature(X_transfer)#h_list[-1] #new_data_entry
        if bytraining:
            #train the last layer with hybrid model
            #train last layer with SVM
            #self.layer_svm = SVC(C=self.C,probability=True)#SVC(kernel='linear', probability=True) #can optimize with parameter
            self.model_finetune = SVC(C=self.C,probability=True) #DecisionTreeClassifier(criterion="entropy", max_depth=1) #LogisticRegression ()
            print('Finetuning with SVM')
            #y = column_or_1d(y_transfer, warn=True)
            strt2=time.time()
            #self.layer_svm.fit(X_finetuned,y_transfer)
            self.model_finetune.fit(X_finetuned,y_transfer)
            print ("Time fitting the data: ",time.time()-strt2)
        #evaluate the finetuning
        #predictions = self.layer_svm.predict(X_finetuned)
        predictions = self.model_finetune.predict(X_finetuned)
        #print (msg," Model accuracy :",accuracy_score(y_transfer,predictions))
        #print (msg," Confusion matrix :\n",confusion_matrix(y_transfer,predictions))
        return predictions
        
    
    def get_last_layer_feature(self,data):
        synthetic_V_list, h_list = self.predict(data)
        X_finetuned = h_list[-1]
        return X_finetuned


    def evaluate(self, X_test,y_test,msg,verbose=False):
        predictions = self.finetune(X_test,y_test,msg, bytraining=False)
        accuracy = accuracy_score(y_test,predictions)
        if verbose:
            print (msg," Model accuracy :",accuracy_score(y_test,predictions))
            print (msg," Confusion matrix :\n",confusion_matrix(y_test,predictions))
        return accuracy
    
    def getweights(self):
        pass



