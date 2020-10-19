# -*- coding: utf-8 -*-

import sys
import numpy as np
from utils import *
from sklearn.model_selection import train_test_split
from RBM import *
from sklearn import metrics #confusion_matrix
#from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
#import seaborn as sns
#%matplotlib inline
class MLSIM(object):
    def __init__(self, input_data=None, label=None,  input_size=2, hidden_layer_sizes=[3, 3], output_size=1,batch_size=10, learning_rate=0.1,epochs=50,C=1,rng=None):
        
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
        self.rng=rng
        #W,hbias,vbias = self.init_weights(input_size,hidden_layer_sizes[0])
        # construct rbm_layer
        rbm_layer = RBM(input=self.input,
                        n_visible=input_size,
                        n_hidden=hidden_layer_sizes[0],
                        batch_size=10,
                        learning_rate=0.1 ,
                        epochs=1000,
                        rng=self.rng)
        self.rbm_layers.append(rbm_layer)
        sample_input =self.input
        for rbm_index in range(self.n_hidden_layers-1):
            _,sample_input =   self.rbm_layers[rbm_index].sample_h_given_v(sample_input)    
            n_hid =  hidden_layer_sizes[rbm_index+1]
            n_vis =  hidden_layer_sizes[rbm_index]
            #W,hbias,vbias = self.init_weights(n_vis,n_hid)
            rbm_i = RBM(input=sample_input,
                        n_visible = n_vis,
                        n_hidden = n_hid,
                        batch_size=10,
                        learning_rate=0.09 ,
                        epochs=100,
                        rng=self.rng)
            self.rbm_layers.append(rbm_i)

  

    def train(self, X_train,save_weigths=False):
        #training
        #print("start the training: ",np.shape(X_train))
        #print("Number of layers ",len(self.rbm_layers))
        dbn_weigths = []
        for rbm_index in range(self.n_hidden_layers):
            rbm =   self.rbm_layers[rbm_index]
            hidden_data_rbm_index = rbm.propup(X_train) #kind prediction last layer 
            hidden_data_rbm_index = np.where(hidden_data_rbm_index>=0.5,1,0)
            print("hidden data shape: ",hidden_data_rbm_index.shape)
            rbm.well_reconstructed(X_train)
            X_train = hidden_data_rbm_index
            #save weights
            if(save_weigths):
                weigths = rbm.getweights()
                dbn_weigths.append(weigths)
        if(save_weigths):
            np.save("dbn_weigths.npy",dbn_weigths)


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
        #h_list = []
        #synthetic_V_list = []
        print("predict---------")
        for rbm in self.rbm_layers:
            #print ("shape Xpredict ",np.shape(X_predict))
            hidden_data_rbm = rbm.propup(X_predict) #rbm.sample_h_given_v(X_predict)
            rbm.well_reconstructed(X_predict)
            #print ("shape hvar ",np.shape(hvar_sample))
            #vvar_mean,vvar_sample = rbm.sample_v_given_h(hvar_sample)
            #print ("shape vvar ",np.shape(vvar_sample))
            X_predict = hidden_data_rbm
            #h_list.append(hvar_mean)
            #synthetic_V_list.append(vvar_mean)
        #print ("shape last layer: ", h_list[0].shape,h_list[1].shape)
        #X_predict = np.where(X_predict>0.5,1,-0.5)
        return X_predict #synthetic_V_list, h_list #h_list[-1] est la dernière couche du Réseau de RBM à utiliser pour finetuner

    def finetune(self,X_transfer, y_transfer,msg, bytraining=True):
        #synthetic_V_list, h_list = self.predict(X_transfer)
        #X_finetuned = self.get_last_layer_feature(X_transfer)#h_list[-1] #new_data_entry
        if bytraining:
            #train the last layer with hybrid model
            #train last layer with SVM
            #self.layer_svm = SVC(C=self.C,probability=True)#SVC(kernel='linear', probability=True) #can optimize with parameter
            self.model_finetune = SVC(gamma='auto',probability=True) #DecisionTreeClassifier(criterion="entropy", max_depth=1) #LogisticRegression ()
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
        accuracy = metrics.accuracy_score(y_test,predictions)
        if verbose:
            print (msg," Model accuracy :",metrics.accuracy_score(y_test,predictions))
            print (msg," Confusion matrix :\n",metrics.confusion_matrix(y_test,predictions))
        return accuracy

    def evaluate2(self, predictions,y_test,msg,verbose=False):
        #predictions = self.finetune(X_test,y_test,msg, bytraining=False)
        accuracy = metrics.accuracy_score(y_test,predictions)
        if verbose:
            print (msg," Model accuracy :",metrics.accuracy_score(y_test,predictions))
            print (msg," Confusion matrix :\n",metrics.confusion_matrix(y_test,predictions))
        return accuracy
    
    def getweights(self):
        pass

def plot_roc_curves(predictions, targets):
    fpr, tpr, _ = metrics.roc_curve(targets,  predictions)
    auc = metrics.roc_auc_score(targets, predictions)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()

def test_dbn():
    print('Start.....')
    data = np.load("/Users/maxwell/Documents/MLSIM/filt_Processing.npy")
    y = np.load("/Users/maxwell/Documents/MLSIM/label_Processing.npy")
    y=np.where(y==-1,0,1)
    yp = np.zeros((data.shape[0],2))
    ind = np.where(y==2)
    yp[ind,0]=1
    ind = np.where(y==1)
    yp[ind,1]=1
    print(yp[0:10])
    rng = np.random.RandomState(123)        
    data_train, data_test, Y_train, Y_test  = train_test_split(data,y, test_size=0.2,random_state=rng) 
    print("shape train ",data_train.shape)
    print("shape test ",data_test.shape)
    #Construct DBN
    mdbn = MLSIM(input_data=data_train,label=Y_train, input_size=data_train.shape[1], hidden_layer_sizes=[7], rng=rng)    
    #------------ greedy training of RBMs --------------
    mdbn.train(data_train)
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(),LogisticRegression(solver='saga',penalty='l2', C=1,verbose=1))#SVC(C=.5, kernel='linear',probability=True))
    #------------ get features of last layers enriched features from RBMs ----------
    dtrain_finetune = mdbn.predict(data_train)
    print("Data to finetuned: ",dtrain_finetune[0:3])
    #----------- Finetune data with the second model -------------------
    clf.fit(data_train, Y_train)
    #----------- Evaluate training data --------------------
    predictions_train = clf.predict(data_train)
    mdbn.evaluate2(predictions_train,Y_train,"Performance MLSIM Training ",verbose=True)
    #----------- Evaluate test data --------------------
    dtest_finetune= mdbn.predict(data_test)
    predictions = clf.predict(data_test)
    predictions_prob = clf.predict_proba(data_test)
    print("predictions proba: ",predictions_prob[0:7])
    mdbn.evaluate2(predictions,Y_test,"Performance MLSIM Test ",verbose=True)
    plot_roc_curves(np.argmax(predictions_prob,axis=1),Y_test)

def test_dbn2():
    print('Start.....')
    data = np.load("/Users/maxwell/Documents/MLSIM/data/phili2017/features/origin/ar.npy")
    y = np.load("/Users/maxwell/Documents/MLSIM/data/phili2017/labels/labels_ar.npy")
    data = data[:,0:3]
    y=np.where(y==-1,0,1)
    yp = np.zeros((data.shape[0],2))
    ind = np.where(y==2)
    yp[ind,0]=1
    ind = np.where(y==1)
    yp[ind,1]=1
    print(yp[0:10])
    rng = np.random.RandomState(123)        
    data_train, data_test, Y_train, Y_test  = train_test_split(data,y, test_size=0.2,random_state=rng) 
    print("shape train ",data_train.shape)
    print("shape test ",data_test.shape)
    print ("Max values ",np.max(data))
    print("25th percentile of data : ",np.percentile(data,25))
    print("50th percentile of data : ",np.percentile(data,50))
    print("75th percentile of data : ",np.percentile(data,75))
    print ("Min values ",np.min(data))
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import normalize
    #Construct DBN
    mdbn = MLSIM(input_data=data_train,label=Y_train, input_size=data_train.shape[1], hidden_layer_sizes=[7], rng=rng)
    clf = make_pipeline(LogisticRegression(solver='saga',C=10,verbose=1))#SVC(C=.5, kernel='linear',probability=True))
    #----------- Finetune data with the second model -------------------
    clf.fit(normalize(data_train), Y_train)
    #----------- Evaluate training data --------------------
    predictions_train = clf.predict(data_train)
    mdbn.evaluate2(predictions_train,Y_train,"Performance MLSIM Training ",verbose=True)
    #----------- Evaluate test data --------------------
    dtest_finetune= mdbn.predict(data_test)
    predictions = clf.predict(normalize(data_test))
    predictions_prob = clf.predict_proba(normalize(data_test))
    print("predictions proba: ",predictions_prob[0:7])
    mdbn.evaluate2(predictions,Y_test,"Performance MLSIM Test ",verbose=True)
    plot_roc_curves(np.argmax(predictions_prob,axis=1),Y_test)

if __name__ == "__main__":
    test_dbn()