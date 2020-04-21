"""
@maxwell.ndognkong
Dec, 7th, 2019
"""
import time
import argparse
import pprint
import numpy as np 
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from model.MDBN import MDBN
from parameters import HYPERPARAMS, OPTIMIZER, DATASET,TRAINING
from preprocess import *

# define the search space
fspace = {
    'optimizer': hp.choice('optimizer', OPTIMIZER.optimizer),
    'optimizer_param': hp.uniform('optimizer_param', OPTIMIZER.optimizer_param['min'], OPTIMIZER.optimizer_param['max']),
    'learning_rate': hp.uniform('learning_rate', OPTIMIZER.learning_rate['min'], OPTIMIZER.learning_rate['max']),    
    'nb_hiden_node1': scope.int(hp.quniform('nb_hiden_node1', OPTIMIZER.nb_hiden_node1['min'], OPTIMIZER.nb_hiden_node1['max'], q=1)),
    'nb_hiden_node2': scope.int(hp.quniform('nb_hiden_node2', OPTIMIZER.nb_hiden_node2['min'], OPTIMIZER.nb_hiden_node2['max'],  q=1)),    
    'epochs': scope.int(hp.quniform('epochs', OPTIMIZER.epochs['min'], OPTIMIZER.epochs['max'],  q=5)), 
    'batch_size': scope.int(hp.quniform('batch_size', OPTIMIZER.batch_size['min'], OPTIMIZER.batch_size['max'], q=5)),  
    'C': hp.uniform('C', OPTIMIZER.C['min'], OPTIMIZER.C['max'])
}

pprint.pprint(fspace["nb_hiden_node1"])

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--max_evals", required=True, help="Maximum number of evaluations during hyperparameters search")
args = parser.parse_args()
max_evals = int(args.max_evals)
current_eval = 1
train_history = []

data =  np.load(DATASET.path_data2[DATASET.LS]["data"])
target = np.load(DATASET.path_data2[DATASET.LS]["label"])
if (np.shape(target)!=2):
    target = np.expand_dims(target,1)
print("shape data ",np.shape(data))
print("shape target ",np.shape(target))
X_train, X_test, y_train, y_test = split_data(data, target,TRAINING.test_percent)
n = np.shape(X_train)[0]
m = np.shape(X_train)[1]
# defint the fucntion to minimize (will train the model using the specified hyperparameters)
def function_to_minimize(hyperparams, optimizer=HYPERPARAMS.optimizer, optimizer_param=HYPERPARAMS.optimizer_param, 
        nb_hiden_node1=HYPERPARAMS.nb_hiden_node1, nb_hiden_node2=HYPERPARAMS.nb_hiden_node2, 
        learning_rate=HYPERPARAMS.learning_rate,
        batch_size=HYPERPARAMS.batch_size,
        epochs=HYPERPARAMS.epochs,
        C=HYPERPARAMS.C):
    if 'learning_rate' in hyperparams: 
        learning_rate = hyperparams['learning_rate']
    if 'batch_size' in hyperparams: 
        batch_size = hyperparams['batch_size']
    if 'nb_hiden_node1' in hyperparams: 
        nhid_param1 = hyperparams['nb_hiden_node1']
    if 'nb_hiden_node2' in hyperparams: 
        nhid_param2 = hyperparams['nb_hiden_node2']   
    if 'C' in hyperparams: 
        C = hyperparams['C'] 
    if 'epochs' in hyperparams: 
        epochs = hyperparams['epochs']        
    if 'optimizer' in hyperparams:
        optimizer = hyperparams['optimizer']
    if 'optimizer_param' in hyperparams:
        optimizer_param = hyperparams['optimizer_param']
    global current_eval 
    global max_evals
    print( "#################################")
    print( "       Evaluation {} of {}".format(current_eval, max_evals))
    print( "#################################")
    start_time = time.time()
    try:
        dbn = MDBN(input_data=X_train, label=y_train, input_size=m, hidden_layer_sizes=[nhid_param1, nhid_param2], output_size=1, batch_size=batch_size, learning_rate=learning_rate,epochs=epochs, C=C)
        start_time = time.time()
        print ("Training the Hybrid Deep Belief Net model \n................")
        #Train the RBM
        dbn.train(X_train)
        #Fine tuning training SVM
        dbn.finetune(X_train,y_train,"finetuning Train data",bytraining=True)
        accuracy = dbn.evaluate(X_train,y_train,"finetuning data",verbose=False)
        #accuracy = len(predictions == validt)/len(predictions)
        training_time = int(round(time.time() - start_time))
        current_eval += 1
        train_history.append({'accuracy':accuracy, 'learning_rate':learning_rate, 'nb_hiden_node1':nhid_param1, 'c':nhid_param2, 'batch_size':batch_size,'epochs':epochs,'C':C,
                                  'optimizer':optimizer, 'optimizer_param':optimizer_param, 'time':training_time})
    except Exception as e:
        # exception occured during training, saving history and stopping the operation
        print( "#################################")
        print( "Exception during training: {}".format(str(e)))
        print( "Saving train history in train_history.npy")
        np.save("train_history.npy", train_history)
        exit()
    return {'loss': -accuracy, 'time': training_time, 'status': STATUS_OK}

# lancer la recherche des  hyperparametres
trials = Trials()
best_trial = fmin(fn=function_to_minimize, space=fspace, algo=tpe.suggest, max_evals=max_evals, trials=trials)

# get some additional information and print the best parameters
for trial in trials.trials:
    if trial['misc']['vals']['learning_rate'][0] == best_trial['learning_rate'] and \
            trial['misc']['vals']['nb_hiden_node1'][0] == best_trial['nb_hiden_node1'] and \
            trial['misc']['vals']['nb_hiden_node2'][0] == best_trial['nb_hiden_node2'] and \
            trial['misc']['vals']['batch_size'][0] == best_trial['batch_size'] and \
            trial['misc']['vals']['epochs'][0] == best_trial['epochs'] and \
            trial['misc']['vals']['C'][0] == best_trial['C'] :
        best_trial['accuracy'] = -trial['result']['loss'] * 100
        best_trial['time'] = trial['result']['time']
print( "#################################")
print( "      Best parameters found")
print( "#################################")
pprint.pprint(best_trial)
print( "#################################")