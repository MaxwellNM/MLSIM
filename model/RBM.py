# -*- coding: utf-8 -*-

import sys
import numpy as np
from utils import *
from sklearn.model_selection import train_test_split
class RBM(object):
    def __init__(self, input=None, n_visible=2, n_hidden=3, \
        W=None, hbias=None, vbias=None, rng=None,\
        learning_rate=None,batch_size=None,epochs=None):
        
        self.n_visible = n_visible  # num of units in visible (input) layer
        self.n_hidden = n_hidden    # num of units in hidden layer

        if rng is None:
            rng = np.random.RandomState(1234)
        if learning_rate is None:
            learning_rate = 0.1
        if batch_size is None:
            batch_size = 10            
        if epochs is None:
            epochs = 1000
        if W is None:
            a = 1. / n_visible
            initial_W = np.array(rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(n_visible, n_hidden)))

            W = initial_W

        if hbias is None:
            hbias = np.zeros(n_hidden)  # initialize h bias 0

        if vbias is None:
            vbias = np.zeros(n_visible)  # initialize v bias 0


        self.rng = rng
        self.input = input
        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.learning_rate =learning_rate
        self.batch_size = batch_size
        self.epochs = epochs


    def contrastive_divergence(self, lr=0.001, k=1, input=None):
        if input is not None:
            self.input = input
        
        ''' CD-k '''
        ph_mean, ph_sample = self.sample_h_given_v(self.input)

        chain_start = ph_sample

        for step in range(k):
            if step == 0:
                nv_means, nv_samples,\
                nh_means, nh_samples = self.gibbs_hvh(chain_start)
            else:
                nv_means, nv_samples,\
                nh_means, nh_samples = self.gibbs_hvh(nh_samples)

        # chain_end = nv_samples


        self.W += lr * (np.dot(self.input.T, ph_mean)
                        - np.dot(nv_samples.T, nh_means))
        self.vbias += lr * np.mean(self.input - nv_samples, axis=0)
        self.hbias += lr * np.mean(ph_mean - nh_means, axis=0)

        cost = self.get_reconstruction_cross_entropy()
        print ('Cost of reconstruction ',cost, 'weigths: ',frobinus_norm(self.W), 'hbias: ',frobinus_norm(self.hbias), 'vbias: ',frobinus_norm(self.vbias))
        # return cost
        
        
    def batch_contrastive_divergence(self, lr=0.1, k=1, input=None, batch_size=20):
        if input is not None:
           self.input = input
        _data = self.input
		#shuffle
        idx = np.random.permutation(len(_data))
        data = _data[idx]  
        n_data =  len(data)     
        n_batches = n_data/batch_size + (0 if n_data % batch_size == 0 else 1)
        for b in range(int(n_batches)):
            batch_x = data[b * batch_size:(b + 1) * batch_size]
            #''' CD-k '''
            ph_mean, ph_sample = self.sample_h_given_v(batch_x)
            
            chain_start = ph_sample
            
            for step in range(k):
                if step == 0:
                   nv_means, nv_samples,\
                   nh_means, nh_samples = self.gibbs_hvh(chain_start)
                else:
                    nv_means, nv_samples,\
                    nh_means, nh_samples = self.gibbs_hvh(nh_samples)
			
            # chain_end = nv_samples
			
			
            self.W += lr * (np.dot(batch_x.T, ph_mean)
							- np.dot(nv_samples.T, nh_means))
            self.vbias += lr * np.mean(batch_x - nv_samples, axis=0)
            self.hbias += lr * np.mean(ph_mean - nh_means, axis=0)
			
            cost = self.batch_get_reconstruction_cross_entropy(batch_x)
            #print ('Cost of reconstruction ',cost, 'weigths: ',frobinus_norm(self.W), 'hbias: ',frobinus_norm(self.hbias), 'vbias: ',frobinus_norm(self.vbias))
			# return cost

    def sample_h_given_v(self, v0_sample):
        h1_mean = self.propup(v0_sample)
        h1_sample = self.rng.binomial(size=h1_mean.shape,   # discrete: binomial
                                       n=1,
                                       p=h1_mean)

        return [h1_mean, h1_sample]


    def sample_v_given_h(self, h0_sample):
        v1_mean = self.propdown(h0_sample)
        v1_sample = self.rng.binomial(size=v1_mean.shape,   # discrete: binomial
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
        cross_entropy =  - np.mean(
            np.sum(self.input * np.log(sigmoid_activation_v) +
            (1 - self.input) * np.log(1 - sigmoid_activation_v),
                      axis=1))
        
        return cross_entropy
        
    def batch_get_reconstruction_cross_entropy(self,batchX):
        pre_sigmoid_activation_h = np.dot(batchX, self.W) + self.hbias
        sigmoid_activation_h = sigmoid(pre_sigmoid_activation_h)
        
        pre_sigmoid_activation_v = np.dot(sigmoid_activation_h, self.W.T) + self.vbias
        sigmoid_activation_v = sigmoid(pre_sigmoid_activation_v)
        #print('Activation ',sigmoid_activation_v)
        #print('Input ',self.input)
        cross_entropy =  - np.mean(
            np.sum(batchX * np.log(sigmoid_activation_v) +
            (1 - batchX) * np.log(1 - sigmoid_activation_v),
                      axis=1))
        
        return cross_entropy

    def reconstruct(self, v):
        print("Test data: \n",v[0:5])
        h = sigmoid(np.dot(v, self.W) + self.hbias)
        reconstructed_v = sigmoid(np.dot(h, self.W.T) + self.vbias)
        a = np.where(reconstructed_v>=0.5,1,0)
        print("Reconstructed Test data: \n",a[0:5])
        dif = np.sqrt(np.sum((v-a)**2,axis=1))
        print ("pattern well classified: \n",np.sum(dif==0,axis=0))
        return reconstructed_v
    
    def well_reconstructed(self,v):
        h = sigmoid(np.dot(v, self.W) + self.hbias)
        reconstructed_v = sigmoid(np.dot(h, self.W.T) + self.vbias)
        a = np.where(reconstructed_v>=0.5,1,0)
        dif = np.sqrt(np.sum((v-a)**2,axis=1))
        wellclassed = np.sum(dif==0,axis=0)
        print ("pattern well classified: \n",wellclassed)
        return wellclassed       

    def getweights(self):
        rbm_weigths = dict()
        rbm_weigths['W1'] = self.W
        rbm_weigths['v1'] = self.vbias
        rbm_weigths['h1'] = self.hbias
        return rbm_weigths

    def load_weights(self,rbm_weigths):
        self.W = rbm_weigths['W1']
        self.vbias = rbm_weigths['v1']
        self.hbias =rbm_weigths['h1']


    def train_rbm(self, data_train, k=1):
        # train
        for epoch in range(self.epochs):
            self.batch_contrastive_divergence(lr=self.learning_rate, k=k, input=data_train, batch_size=self.batch_size)
            # cost = rbm.get_reconstruction_cross_entropy()
            # print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, cost
        weithd = self.getweights()
        np.save("weigths_rbm", weithd)
        np.savetxt("weigths_W", weithd['W1'])

def frobinus_norm(A):
    """
    Generate matrix norm of Frobenius
    :param data: array-like, weights of a hidden layer
    :return:
    """   
    return np.sqrt(np.sum(A**2)) 

def test_rbm(learning_rate=0.1, k=1, training_epochs=1000,batch_size=20):
    data = np.array([[1,1,1,0,0,0],
                        [1,0,1,0,0,0],
                        [1,1,1,0,0,0],
                        [0,0,1,1,1,0],
                        [0,0,1,1,0,0],
                        [0,0,1,1,1,0]])
    data = np.load("/Users/maxwell/Documents/MLSIM/ls_Processing.npy")
    y = np.load("/Users/maxwell/Documents/MLSIM/label_Processing.npy")
    rng = np.random.RandomState(123)        
    data_train, data_test, Y_train, Y_test  = train_test_split(data,y, test_size=0.2,random_state=rng) 
    print("shape train ",data_train.shape)
    print("shape test ",data_test.shape)

    #Construct RBM
    rbm = RBM(n_visible=data_train.shape[1], n_hidden=9, rng=rng)
    #Train
    rbm.train_rbm(data_train, k=1)
    rbm.well_reconstructed(data_train)

    # test
    v = np.array([[1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 0]])

    rbm.reconstruct(data_test)

    print("Weigths: \n")
    print (rbm.W)
    print("hbias; \n")
    print(rbm.hbias)
    print("vbias; \n")
    print(rbm.vbias)	



if __name__ == "__main__":
    test_rbm()
