from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
from parameters import *
from utils import *
import RBM
import time
import os
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
class MDBN2(object):
    def __init__(self, input_data=None, label=None,  input_size=2, hidden_layer_sizes=[3, 3], output_size=1,batch_size=TRAINING.batch_size, learning_rate=0.1,epochs=50):
        
        self.input = input_data.astype(np.float32)
        self.y = label.astype(np.float32)
        self.rbm_layers = []
        self.n_hidden_layers = len(hidden_layer_sizes)  # nombre de couches cachées du DBN
        self.batch_size = batch_size
        self.layer_svm = None
        self.nepochs = epochs
        self.learning_rates = learning_rate
        W,hbias,vbias = self.init_weights(input_size,hidden_layer_sizes[0])
        # construct rbm_layer
        rbm_layer = RBM(input=self.input,
                        n_visible=input_size,
                        n_hidden=hidden_layer_sizes[0],
                        W=W,  
                        hbias=hbias,
                        vbias=vbias,
                        batch_size=TRAINING.batch_size)
        self.rbm_layers.append(rbm_layer)
        for rbm_index in range(1,self.n_hidden_layers-1):
            sample_input =   self.rbm_layers[rbm_index-1].sample_h_given_v()    
            n_hid =   hidden_layer_sizes[rbm_index]
            W,hbias,vbias = self.init_weights(input_size,n_hid)
            rbm_i = RBM(input=sample_input,
                        n_visible = input_size,
                        n_hidden = n_hid,
                        W=W,  
                        hbias=hbias,
                        vbias=vbias,
                        batch_size=TRAINING.batch_size)
            self.rbm_layers.append(rbm_i)

  

    def train(self, X_train):
        #training
        for rbm_index in range(self.n_hidden_layers):
            rbm =   self.rbm_layers[rbm_index]
            rbm._stochastic_gradient_descent(X_train)
            X_train = rbm.predict(X_train)
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
                size=(n_visible, number_neuron_hiden)))

            W = initial_W

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
            hvar = rbm.sample_h_given_v(X_predict)
            vvar = rbm.sample_v_given_h(X_predict)
            #synthetic_v, X_predict = rbm.predict(X_predict)
            h_list.append(hvar)
            synthetic_V_list.append(vvar)

        return synthetic_V_list, h_list #h_list[-1] is the last layer of the RBM stack

    def finetune(self,X_transfer, bytraining=False):
        synthetic_V_list, h_list = self.predict(X_transfer)
        new_data_entry = h_list[-1]
        if bytraining:
            pass
            #train the last layer with hybrid model
        else:
            pass


    def predict_class(self,X_test):
        pass
    
    def getweights(self):
        pass




