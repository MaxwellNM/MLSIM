# -*- coding: utf-8 -*-

import sys
import numpy as np
from utils import *
from parameters import TRAINING



class LogisticRegression(object):
    def __init__(self, input, label, n_in, n_out):
        self.x = input
        self.y = label
        a = 1. / n_in
        initial_W = np.array(np.random.uniform(  # initialize W uniformly
            low=-a,
            high=a,
            size=(n_in, n_out)))
        self.W = initial_W #np.zeros((n_in, n_out))  # initialize W 0
        print (self.W)
        self.b = np.zeros(n_out)  # initialize bias 0


    def train(self, lr=0.1, input=None, L2_reg=0.01):        
        if input is not None:
            self.x = input

        p_y_given_x = self.output(self.x)
        d_y = self.y - p_y_given_x

        self.W += lr * np.dot(self.x.T, d_y) - lr * L2_reg * self.W
        self.b += lr * np.mean(d_y, axis=0)
        self.d_y = d_y


    def _stochastic_gradient_descent(self,lr, _data):
        """
        Performs stochastic gradient descend optimization algorithm.
        :param _data: array-like, shape = (n_samples, n_features)
        :return:
        """
        accum_delta_W = np.zeros(self.W.shape)
        accum_delta_b = np.zeros(self.b.shape)
        train_history = []
        for iteration in range(1, TRAINING.iteration_last_layer + 1):
            idx = np.random.permutation(len(_data))
            data = _data[idx]
            y = self.y[idx]
            #add y to permute the data
            data2 = np.concatenate((data,y),axis=1)
            print(data2.shape)
            for batch in batch_generator(TRAINING.batch_size, data2):
                #accum_delta_W[:] = .0
                #accum_delta_b[:] = .0
                # for sample in batch:
                #     sample_X = sample[:-1]
                #     sample_y = sample[-1]
                #     sample_X = np.expand_dims(sample_X,1)
                p_y_given_x = self.output(batch[:,:-1])
                #print("shape p_y_given_x ; ", np.shape(p_y_given_x))
                y_batch = np.expand_dims(batch[:,-1],1)
                d_y = y_batch - p_y_given_x  #error   
                error = np.mean(d_y**2) #self.negative_log_likelihood(batch[:,:-1],y_batch)
                #print("shape dy ; ", np.shape(d_y)," batch labels: ",np.shape(y_batch))            
                #delta_W, delta_b = self.contrastive_divergence(self, k=HYPERPARAMS.constrative_k,input=sample)
                delta_W= self.W*error
                delta_b = self.b*error
                accum_delta_W += delta_W
                accum_delta_b += delta_b
                self.W += lr * (accum_delta_W / TRAINING.batch_size)
                self.b += lr* (accum_delta_b / TRAINING.batch_size)
            if TRAINING.save_history:
                error = self.negative_log_likelihood(data,y) #_compute_reconstruction_error(data)
                train_history.append({ 'epoch Logistic':iteration, 'cost':error})
                print(">> Epoch %d finished \t Logistic cost error %f" % (iteration, error))
            lr *= 1/(iteration+1)
        return train_history
        

    # def train(self, lr=0.1, input=None, L2_reg=0.00):
    #     self.forward(input)
    #     self.backward(lr, L2_reg)

    # def forward(self, input=None):
    #     if input is not None:
    #         self.x = input

    #     p_y_given_x = self.output(self.x)
    #     self.d_y = self.y - p_y_given_x
        
    # def backward(self, lr=0.1, L2_reg=0.00):
    #     self.W += lr * np.dot(self.x.T, self.d_y) - lr * L2_reg * self.W
    #     self.b += lr * np.mean(self.d_y, axis=0)


    def output(self, x):
        #print("In Logistic output: ",np.shape(x)," and  ",np.shape(self.W),np.shape(self.b))
        return sigmoid(np.dot(x, self.W) + self.b)
        #return softmax(np.dot(x, self.W) + self.b)

    def predict(self, x):
        return self.output(x)


    def negative_log_likelihood(self,X,Y):
        # sigmoid_activation = sigmoid(np.dot(self.x, self.W) + self.b)
        sigmoid_activation = sigmoid(np.dot(X, self.W) + self.b)#softmax()
        #print ('min sigmoid: ',np.min(sigmoid_activation),'max sigmoid: ',np.max(sigmoid_activation))
        #sigmoid_activation = np.where(sigmoid_activation>=0.4, 1, 0)
        unique, counts = np.unique(sigmoid_activation, return_counts=True)
        unique2, counts2 = np.unique(self.y, return_counts=True)
        #print("activation: ",dict(zip(unique,counts)))
        #print("shape y: ",dict(zip(unique2,counts2)))
        #print('Shape y: ',self.y.shape)
        cross_entropy = - np.mean(
            np.sum(Y * np.log(sigmoid_activation) +
            (1 - Y) * np.log(1 - sigmoid_activation),
                      axis=1))

        return cross_entropy


def test_lr(learning_rate=0.1, n_epochs=500):

    rng = np.random.RandomState(123)

    # training data
    d = 2
    N = 10
    x1 = rng.randn(N, d) + np.array([0, 0])
    x2 = rng.randn(N, d) + np.array([20, 10])
    y1 = [[1, 0] for i in xrange(N)]
    y2 = [[0, 1] for i in xrange(N)]

    x = np.r_[x1.astype(int), x2.astype(int)]
    y = np.r_[y1, y2]


    # construct LogisticRegression
    classifier = LogisticRegression(input=x, label=y, n_in=d, n_out=2)

    # train
    for epoch in xrange(n_epochs):
        classifier.train(lr=learning_rate)
        # cost = classifier.negative_log_likelihood()
        # print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, cost
        learning_rate *= 0.995


    # test
    result = classifier.predict(x)
    for i in xrange(N):
        print (result[i])
    print("\n")
    for i in xrange(N):
        print (result[N+i])



if __name__ == "__main__":
    test_lr()
