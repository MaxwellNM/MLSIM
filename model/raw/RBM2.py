# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 17:05:16 2018
@author: Alexandre Boyker
"""
import numpy as np
from tf_helper import reset_graph
import tensorflow as tf
import os
from datetime import datetime
from utils import *

class RBM(object):
    """
    
    this class implements a Restricted Boltzmann Machine (RBM) whose input are p-dimensional
    vectors. In principle, those input vectors are assumed to take value in {0, 1}^p (p-dimensional boolean vectors),
    but in practice this assumption can be violated.
    
    The RBM learns the probability distribution P(h|V; W, b) and P(V|h; W, a)
    
    
            embedding layer (size k<p):    [1, 1, 0,..., 1, 0, 1] 
                                          / |/\/ \/\ \ ...
                                         / /| /\ /\ \ \ ...
            input layer (size p):       [1, 0, 0, 0, 1, 1, 0, ..., 0, 1] 
    
    Naming convention:
        
        - input_size: dimension of the p-dimensional input samples (p) 
        - embedding_size: dimension of the embedded vectors
        - V: input vectors, generally a batch of size (#batches * input_size)
        - h: learnt embedding, boolean k-dimensional vector, generally a batch of size (#batches * embedding_size)
        - W: weight matrix connecting V and h through P(h|V; W, b) and P(V|h; W, a)
        - a, b: bias terms
        
        
    The method used to train the RBM is the contrastive divergence algorithm (approximation of the gradient
    of the log-likelihood).
    
    https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine
    
    """
    def __init__(self, input_size=12 , embedding_size=10, batch_size=20, learning_rate= 0.1, n_epochs=2, saved_model_directory="saved_models", model_id=0):
        
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.model_id = model_id
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.saved_model_directory = os.path.join(saved_model_directory, "hidden_layer_" + str(self.model_id))
    

    def get_model_size(self):
        
        return self.input_size, self.embedding_size
    
    def _p_v_given_h(self, W, h, a):
    
        """
        
        Returns the conditional probability P(h|V; W, b) as a tensor object
        
        positional arguments:
            
            -- W: weights matrix, tensor object
            -- h: (batch) of embedding vectors, tensor object
            -- a: bias term, tensor object
        
        """ 
        return sigmoid(np.dot(h,W.T)+a)
        #return tf.sigmoid(tf.add(a, tf.matmul(h,tf.transpose(W))))
        
    def _p_h_given_v(self, W, V, b):
        
        """
        Returns the conditional probability P(V|h; W, a) as a tensor object
        of size (Batch_size * input_size)
        
        positional arguments:
            
            -- W: weights matrix, tensor object
            -- V: (batch) of input vectors, tensor object
            -- b: bias term, tensor object
        """

        return sigmoid(np.dot(V,W)+b)
        #return tf.sigmoid(tf.add(b , tf.matmul(V, W) ))
    
    def _sample_h(self, W, V, b):
        
        """
        Returns the sampled embedding h according to the distribution P(h|V; W, b) as a
        tensor of size (batch_size * embedding_size)
        
        positional arguments:
            
            -- W: weights matrix, tensor object
            -- V: (batch of) input vectors, tensor object
            -- b: bias term, tensor object
        
        """
        proba = self._p_h_given_v( W, V, b)
        dist = tf.distributions.Bernoulli(probs=proba, dtype=tf.float32)
        return tf.cast(dist.sample(), tf.float32)
    
    def _sample_V(self, W, h, a):
        
        """
        Returns the reconstructed version of the input V_synthetic according to the distribution P(V|h; W, a) as a
        tensor of size (batch_size * input_size)
        
        positional arguments:
            
            -- W: weights matrix, tensor object
            -- h: (batch of) input vectors, tensor object
            -- a: bias term, tensor object
        
        """
        proba =  self._p_v_given_h(W, h, a)
        dist = tf.distributions.Bernoulli(probs=proba, dtype=tf.float32)

        return tf.cast(dist.sample(), tf.float32)
    
    def _outer_product_batches(self, V, h):
        
        """
        Returns the outer product of batches of vectors.
        The outer product output is a tensor of size (batch_size * input_size * embedding_size)
        
        positional arguments:
            
            -- V: tensor of input vectors of size (batch_size * input_size)
            -- h: tensor of embedding vectors of size (batch_size * embedding_size)
        """
        x_as_matrix_batch = tf.expand_dims(V, 2)
        y_as_matrix_batch = tf.expand_dims(h, 1)

        return tf.matmul(x_as_matrix_batch, y_as_matrix_batch)
        
    
    def _initialize_variables(self):
        
        """
        Returns initial values for W, a, b to feed the algorithm.
        
        """
        b_ = np.random.uniform(low=-1.0, high=1.0, size=(self.embedding_size))
        a_ = np.random.uniform(low=-1.0, high=1.0, size=(self.input_size))
        W_ = np.random.uniform(low=-1.0, high=1.0, size=(self.input_size,self.embedding_size))
  
        return W_, a_, b_
    
    def _build_graph(self):
        
        """
        
        Returns key elements of the tensorflow computational graph. 
        
        Training is done by "manually" updating W, a and b using contrasting divergence.
        
        
        """
        
        b_var = tf.Variable(tf.random_uniform([self.embedding_size], -1.0, 1.0))
        a_var = tf.Variable(tf.random_uniform([self.input_size], -1.0, 1.0))
        W_var = tf.Variable(tf.random_uniform([self.input_size, self.embedding_size], -1.0, 1.0) )
        
        V_plac = tf.placeholder(tf.float32, [None, self.input_size], name="V_plac")
        W_plac = tf.placeholder(tf.float32, [self.input_size, self.embedding_size], name="W_plac")
        a_plac = tf.placeholder(tf.float32, [self.input_size], name="a_plac")
        b_plac = tf.placeholder(tf.float32, [self.embedding_size], name="b_plac")
        
        h_var = self._sample_h(W_plac, V_plac, b_plac)

        positive_outer = self._outer_product_batches(V_plac, h_var)
        
        synthetic_V =  self._sample_V(W_plac, h_var, a_plac)
        synthetic_h = self._sample_h(W_plac, synthetic_V, b_plac)
        negative_outer = self._outer_product_batches(synthetic_V, synthetic_h)
        
        
        W_gradient = tf.reduce_mean(tf.add(positive_outer, - negative_outer), axis=0)
        b_gradient =  tf.reduce_mean(h_var - synthetic_h, axis=0)
        a_gradient =  tf.reduce_mean( V_plac - synthetic_V, axis=0)
      
        W_var = tf.assign(W_var, W_var + self.learning_rate * W_gradient)
        b_var = tf.assign(b_var, b_var + self.learning_rate * b_gradient)
        a_var = tf.assign(a_var, a_var + self.learning_rate * a_gradient)
        
        return V_plac, W_plac, a_plac, b_plac, W_var, a_var, b_var, W_gradient, b_gradient, a_gradient, positive_outer, negative_outer, synthetic_V, synthetic_h, h_var
        
    
    def _save_parameters(self, W, a, b):
        
        """
        Save RBM weights.
        
        positional arguments:
            
            -- W: weight matrix, nupy ndarray of size (input_size * embedding_size)
            -- a: bias term, numpy array of size (input_size)
            -- b: bias term, numpy array of size (embedding_size)
       
        """
        
        if not os.path.exists(self.saved_model_directory):
            os.makedirs(self.saved_model_directory)
            
        np.save(os.path.join(self.saved_model_directory, "W"), W)
        np.save(os.path.join(self.saved_model_directory, "a"), a)
        np.save(os.path.join(self.saved_model_directory, "b"), b)
    
    def _build_prediction_graph(self):
        
        """
        
        Returns graphs elements to make predictions on an RBM.
        
        """
        V_plac = tf.placeholder(tf.float32, [self.batch_size, self.input_size], name="V_plac")
        W_plac = tf.placeholder(tf.float32, [self.input_size, self.embedding_size], name="W_plac")
        b_plac = tf.placeholder(tf.float32, [self.embedding_size], name="b_plac")
        a_plac = tf.placeholder(tf.float32, [ self.input_size], name="a_plac")

        h_pred = self._sample_h(W_plac, V_plac,b_plac)
        synthetic_V =  self._sample_V(W_plac, h_pred, a_plac)
        
        return V_plac, W_plac, b_plac,a_plac, h_pred, synthetic_V
    
    def predict(self, X_pred, W=None, a=None, b=None):
        
        """
        Returns reconstructed input as well as the learnt embedding as numpy ndarrays
        
        positional arguments:
            
            -- X_pred: numpy ndarray of input. We must have X_pred.shape[1] = embedding_size
        
        keyword arguments:
            
            The keyword arguments are initial weigths for prediction. If the weights are None, they will be 
            loaded from the model directory.
            
            
            -- W: weight matrix, nupy ndarray of size (input_size * embedding_size)
            -- a: bias term, numpy array of size (input_size)
            -- b: bias term, numpy array of size (embedding_size)
        
        """
        reset_graph()
        init = tf.global_variables_initializer()
        
        if ((W is None)  or (b is None) or (a is None)  ):

            W_feed = np.load(os.path.join(self.saved_model_directory, "W.npy"))
            b_feed = np.load(os.path.join(self.saved_model_directory, "b.npy"))
            a_feed = np.load(os.path.join(self.saved_model_directory, "a.npy"))
            
        else:
            
            W_feed = W
            b_feed = b
            a_feed = a
            
        V_plac, W_plac, b_plac,a_plc, h_pred, synthetic_V = self._build_prediction_graph()
        predictions_V = []
        predictions_h = []
        with tf.Session() as sess:

             sess.run(init)
             for batch_number in range(0, X_pred.shape[0], self.batch_size):
                 
                batch = X_pred[batch_number:batch_number + self.batch_size]
                predi_v, predi_h = sess.run([synthetic_V, h_pred],feed_dict ={V_plac: batch, W_plac:W_feed, b_plac:b_feed, a_plc:a_feed})
                predictions_V.append(predi_v)
                predictions_h.append(predi_h)
                
        predictions_V = np.array(predictions_V)
        predictions_h = np.array(predictions_h)
        return predictions_V.reshape(predictions_V.shape[0] * predictions_V.shape[1], predictions_V.shape[2] ), predictions_h.reshape(predictions_h.shape[0] * predictions_h.shape[1], predictions_h.shape[2] )
 
    
    def train(self, X_train):
        
        """
        Train the RBM
        
        positional arguments:
            
            -- X_train: numpy ndarray of size (any integer * input_size)
        
        """
        V_plac, W_plac, a_plac, b_plac, W_var, a_var, b_var, W_gradient, b_gradient, a_gradient, positive_outer, negative_outer,synthetic_V, synthetic_h, h = self._build_graph()
        n_samples = X_train.shape[0]
        init = tf.global_variables_initializer()
        
        W_feed, a_feed, b_feed = self._initialize_variables()
        
        with tf.Session() as sess:

            sess.run(init)
            
            for i in range(self.n_epochs):
                
                for batch_number in range(0, n_samples, self.batch_size):
                    
                    batch = X_train[batch_number:batch_number + self.batch_size]
                    
                
                    W_feed, a_feed, b_feed, W_grad, b_grad, a_grad = sess.run([W_var, a_var, b_var, W_gradient, b_gradient, a_gradient],feed_dict ={V_plac: batch, W_plac:W_feed, a_plac:a_feed, b_plac:b_feed})
                    
                    if batch_number % 1000 == 0:
                        
                        print("{} iterations: {} out of {}  W-error: {}  b-error: {} a-error: {}".format(str(datetime.now()),batch_number +  i * n_samples, self.n_epochs*n_samples, self.learning_rate * np.sum(W_grad**2), self.learning_rate *np.sum(b_grad**2), self.learning_rate * np.sum(a_grad **2)))
                      
                        
            
            self._save_parameters( W_feed, a_feed, b_feed)