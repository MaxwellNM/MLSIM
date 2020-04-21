"""
@maxwell.ndognkong
Dec, 7th, 2019
"""
from extraction import *
from preprocess import *
from parameters import *
from  model.MDBN import MDBN
import numpy as np
import time
import pickle
import argparse
from sklearn.metrics import confusion_matrix, accuracy_score

np.random.seed(1337)  # for reproducibility

#Using moodle 2017 data
def train_evaluate(pretrain_lr=HYPERPARAMS.learning_rate, pretraining_epochs=1500, k=HYPERPARAMS.constrative_k, \
             finetune_lr=HYPERPARAMS.learning_rate, finetune_epochs=200, LS="SG"):
    #load the data
    #print(DATASET.path_data[LS])
    data =  np.load(DATASET.path_data2[LS]["data"])
    target = np.load(DATASET.path_data2[LS]["label"])
    if (np.shape(target)!=2):
        target = np.expand_dims(target,1)
    # Split into training, validation, and test sets
    if TRAINING.test_percent is None:
        train, traint, valid, validt, test, testt = split_data(data, target,TRAINING.test_percent)
    else:
        X_train, X_test, y_train, y_test = split_data(data, target,TRAINING.test_percent)
    n = np.shape(X_train)[0]
    m = np.shape(X_train)[1]
    print (n,m)
    print ("Nbr students:", n)
    print ("Learning traces dimension:",m)
    # construct DBN
    nhid_param2 = HYPERPARAMS.nb_hiden_node2 #nobre de noeuds de la deuxime couche cachée
    nhid_param1 = HYPERPARAMS.nb_hiden_node1 #nobre de noeuds de la premiere couche cachée
    dbn = MDBN(input_data=X_train, label=y_train, input_size=m, hidden_layer_sizes=[nhid_param1, nhid_param2], output_size=1, batch_size=Training.batch_size, learning_rate=HYPERPARAMS.learning_rate,epochs=Training.epochs, C=HYPERPARAMS.C)
    start = time.time()
    # pre-training (TrainUnsupervisedDBN)
    print ("Training the Hybrid Deep Belief Net model \n................")
    dbn.train(X_train)
    print ("Unsupervised training time: ",(time.time()-start)," seconds")
    # fine-tuning (DBNSupervisedFineTuning)
    synthetic_V_list, h_list = dbn.predict(X_train)
    dbn.finetune(X_train,y_train,"Train data",bytraining=True)
    accuracy = dbn.evaluate(X_test,y_test,"Test data",verbose=True)
    if TRAINING.savemodel :
        timestr = time.strftime("%Y%m%d-%H%M%S")
        fold = os.path.join(TRAINING.path_saved_model+os.sep,time.strftime("%Y-%m-%d"))
        if not os.path.isdir(fold):
            os.mkdir(fold)
        fl = fold+os.sep+"mlsim_"+timestr+".pkl"
        model = pickle.dump(dbn,open(fl,"wb"))
        print ('Path of the model: ',fl)
        if TRAINING.evaluate:
            with open(fl, 'rb') as pickle_file:
                model = pickle.load(pickle_file)
                predictions = model.predict(X_test)
                #print(y_test)
                #print(predictions)
                print ("Model accuracy :",accuracy_score(y_test,predictions))
                print ("Confusion matrix :\n",confusion_matrix(y_test,predictions))
                #print (len(predictions == y_test)/len(predictions))
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("-t", "--train", default="yes", help="if 'yes', launch training from command line")
    #parser.add_argument("-e", "--evaluate", default="no", help="if 'yes', launch evaluation on test dataset")
    #parser.add_argument("-ls", "--learning_style", default="AR", help="Learning style dimension to process")
    #args = parser.parse_args()
    #if args.train=="yes" or args.train=="Yes" or args.train=="YES":
    #        train_evaluate()
    #if args.evaluate=="yes" or args.evaluate=="Yes" or args.evaluate=="YES":
    #        train_evaluate(train_model=False)  
    train_evaluate()  
        
