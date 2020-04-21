# -*- coding: utf-8 -*-

import sys
import numpy as np
from .HiddenLayer import HiddenLayer
from .LogisticRegression import LogisticRegression
from parameters import *
from utils import *
import time
import os

"""Cite RBM author class """
"""xx. xxxx,2010 """
class RBM(object):
    def __init__(self, input=None, n_visible=2, n_hidden=3, \
        W=None, hbias=None, vbias=None, randomizer=None):
        
        self.n_visible = n_visible  # num of units in visible (input) layer
        self.n_hidden = n_hidden    # num of units in hidden layer

        if randomizer is None:
            randomizer = np.random.RandomState(1234)


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


        self.randomizer = randomizer
        self.input = input
        self.W = W
        self.hbias = hbias
        self.vbias = vbias


    """Machine Learning: An Algorithmic Perspective, Second Edition https://www.goodreads.com/book/show/20607838-machine-learning
       by Stephen Marsland pp.372-373"""
    def contrastive_divergence(self, lr=0.01, k=1, input=None):
        if input is not None:
            self.input = input

        ''' Constrastive divergence - k '''
        ph_mean, ph_sample = self.sample_h_given_v(self.input)

        chain_start = ph_sample

        for step in range(k):
            if step == 0:
                nv_means, nv_samples,\
                nh_means, nh_samples = self.gibbs_hvh(chain_start)
            else:
                nv_means, nv_samples,\
                nh_means, nh_samples = self.gibbs_hvh(nh_samples)

        #chain_end = nv_samples


        self.W += lr * (np.dot(self.input.T, ph_mean)
                        - np.dot(nv_samples.T, nh_means))
        self.vbias += lr * np.mean(self.input - nv_samples, axis=0)
        self.hbias += lr * np.mean(ph_mean - nh_means, axis=0)
        cost = self.get_reconstruction_cross_entropy()
        # return cost


    def sample_h_given_v(self, v0_sample):
        h1_mean = self.propup(v0_sample)
        h1_sample = self.randomizer.binomial(size=h1_mean.shape,   # discrete: binomial
                                       n=1,
                                       p=h1_mean)

        return [h1_mean, h1_sample]


    def sample_v_given_h(self, h0_sample):
        v1_mean = self.propdown(h0_sample)
        v1_sample = self.randomizer.binomial(size=v1_mean.shape,   # discrete: binomial
                                            n=1,
                                            p=v1_mean)
        
        return [v1_mean, v1_sample]

    def propup(self, v):
        pre_sigmoid_activation = np.dot(v, self.W) + self.hbias
        return sigmoid(pre_sigmoid_activation)

    def propdown(self, h):
        pre_sigmoid_activation = np.dot(h, self.W.T) + self.vbias
        return sigmoid(pre_sigmoid_activation)


    def gibbs_hvh(self, h0_sample):
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        h1_mean, h1_sample = self.sample_h_given_v(v1_sample)

        return [v1_mean, v1_sample,
                h1_mean, h1_sample]
    

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



class myDBN(object):
    def __init__(self, input=None, label=None,\
                 n_ins=2, hidden_layer_sizes=[3, 3], n_outs=2,\
                 randomizer=None):
        
        self.input = input
        self.y = label

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.n_layers = len(hidden_layer_sizes)  # nombre de couches cachées du DBN

        if randomizer is None:
            randomizer = np.random.RandomState(1234)
        assert self.n_layers > 0

        # construct multi-layer
        for i in range(self.n_layers):
            # layer_size
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layer_sizes[i - 1]

            # layer_input
            if i == 0:
                layer_input = self.input
            else:
                layer_input = self.sigmoid_layers[-1].sample_h_given_v()
                
            # construct sigmoid_layer
            sigmoid_layer = HiddenLayer(input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layer_sizes[i],
                                        rng=randomizer,
                                        activation=sigmoid)
            self.sigmoid_layers.append(sigmoid_layer)


            # construct rbm_layer
            rbm_layer = RBM(input=layer_input,
                            n_visible=input_size,
                            n_hidden=hidden_layer_sizes[i],
                            W=sigmoid_layer.W,     # W, b are shared
                            hbias=sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)


        # layer for output using Logistic Regression
        self.log_layer = LogisticRegression(input=self.sigmoid_layers[-1].sample_h_given_v(),
                                            label=self.y,
                                            n_in=hidden_layer_sizes[-1],
                                            n_out=n_outs)

        # finetune cost: the negative log likelihood of the logistic regression layer
        self.finetune_cost = self.log_layer.negative_log_likelihood()        

    def fit(self, lr=0.1, k=1, epochs=100, verbose=False,save_history=False):
        train_history = []        
        # pre-train each RBM
        for i in range(self.n_layers):
            if i == 0:
                layer_input = self.input
            else:
                layer_input = self.sigmoid_layers[i-1].sample_h_given_v(layer_input)
            rbm = self.rbm_layers[i]
            #trainin the RBM
            for epoch in range(epochs):
                rbm.contrastive_divergence(lr=lr, k=k, input=layer_input)
                cost = rbm.get_reconstruction_cross_entropy()
                train_history.append({'RBM':i+1, 'epoch':epoch, 'cost':cost})
                if verbose :
                    print ('Pre-training RBM %d, epoch %d, cost ' %(i+1, epoch), cost)
        if save_history:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            np.save(TRAINING.training_logs+os.sep+"train-"+timestr+".npy" ,train_history)


    def finetune(self, lr=0.1, epochs=100):
        layer_input = self.sigmoid_layers[-1].sample_h_given_v()

        # train log_layer
        epoch = 0
        done_looping  = False
        while (epoch < epochs) and (not done_looping):
            self.log_layer.train(lr=lr, input=layer_input)
            self.finetune_cost = self.log_layer.negative_log_likelihood()
            print ('Training epoch %d, cost is ' % epoch, self.finetune_cost)
            
            lr *= 0.95
            epoch += 1

    def predict(self, x):
        layer_input = x
        
        for i in range(self.n_layers):
            sigmoid_layer = self.sigmoid_layers[i]
            layer_input = sigmoid_layer.output(input=layer_input)

        out = self.log_layer.predict(layer_input)
        return out


