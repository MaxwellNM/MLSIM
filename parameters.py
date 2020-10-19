
class Dataset:
    model_name="MLSMI" # Manga Learning Style Model Identifier
    AR = "Active_Reflective"
    SG = "Sequential_Global"
    VV = "Visual_Verbal"
    LS="SG"
    #path of binarized data
    path_data = {}
    path_data["AR"]  = {'data': "./data/moodle2017/features/binarize/ar.npy", 'label': "./data/moodle2017/labels/label_ar.npy"} #["./data/features/binarize/ar.npy","./data/labels/label_ar.npy"]
    path_data["SG"]  = {'data': "./data/moodle2017/features/binarize/sg.npy", 'label': "./data/moodle2017/labels/label_sg.npy"} #["./data/features/binarize/sg.npy","./data/labels/label_sg.npy"]
    path_data["SI"]  = {'data': "./data/moodle2017/features/binarize/si.npy", 'label': "./data/moodle2017/labels/label_si.npy"} #["./data/features/binarize/si.npy","./data/labels/label_si.npy"]
    path_data["VV"]  = {'data': "./data/moodle2017/features/binarize/vv.npy", 'label': "./data/moodle2017/labels/label_vv.npy"} #["./data/features/binarize/vv.npy","./data/labels/label_vv.npy"]
    path_data2 = {}
    path_data2["AR"]  = {'data': "./data/phili2017/features/binarize/ar.npy", 'label': "./data/phili2017/labels/labels_ar.npy"} #["./data/features/binarize/ar.npy","./data/labels/label_ar.npy"]
    path_data2["SG"]  = {'data': "./data/phili2017/features/binarize/sg.npy", 'label': "./data/phili2017/labels/labels_sg.npy"} #["./data/features/binarize/sg.npy","./data/labels/label_sg.npy"]
    path_data2["SI"]  = {'data': "./data/phili2017/features/binarize/si.npy", 'label': "./data/phili2017/labels/labels_si.npy"} #["./data/features/binarize/si.npy","./data/labels/label_si.npy"]
    path_data2["VV"]  = {'data': "./data/phili2017/features/binarize/vv.npy", 'label': "./data/phili2017/labels/labels_vv.npy"} #["./data/features/binarize/vv.npy","./data/labels/label_vv.npy"]
   
    #otherdata = dict()

class Training:
    training_logs= "./logs/trainedRBM"
    save_history =True
    savemodel = True
    evaluate = True
    epochs = 100 #nbre iteration RBM
    batch_size = 50       
    iteration_last_layer = 100 #Iteration derniere couche
    test_percent = 0.2
    path_saved_model ="./output"
# {'C': 7.682409393919036,
#  'accuracy': 90.13840830449827,
#  'batch_size': 45.0,
#  'epochs': 90.0,
#  'learning_rate': 0.18888830446130184,
#  'nb_hiden_node1': 6.0,
#  'nb_hiden_node2': 6.0,
#  'optimizer': 0,
#  'optimizer_param': 0.7927236881225321,
#  'time': 58}

# #################################
# {'C': 0.29775344838807594,
#  'accuracy': 71.85185185185186,
#  'batch_size': 25.0,
#  'epochs': 375.0,
#  'learning_rate': 0.3943414157393338,
#  'nb_hiden_node1': 8.0,
#  'nb_hiden_node2': 9.0,
#  'optimizer': 0,
#  'optimizer_param': 0.6111803921294505,
#  'time': 58}
# #################################

#SG
# #################################
# {'C': 0.28346943095940424,
#  'accuracy': 80.98765432098766,
#  'batch_size': 10.0,
#  'epochs': 50.0,
#  'learning_rate': 0.35198141735557187,
#  'nb_hiden_node1': 5.0,
#  'nb_hiden_node2': 6.0,
#  'optimizer': 0,
#  'optimizer_param': 0.5788442901457271,
#  'time': 8}
# #################################

class HYPERPARAMS:
    #RBM Section
    constrative_k = 1   # 
    learning_rate = 0.3#0.3943#0.188
    nb_hiden_node1 = 10
    nb_hiden_node2 = 10
    C = 1#0.2977 #SVM parameter
    epochs = 500 #nbre iteration RBM
    batch_size = 32    
    optimizer_logs = "./logs/optimizeRBM"
    #DBN section
    #nb_layers = 1   # dropout = 1 - keep_prob
    optimizer = 'skopt'  # {'bayesian', 'skopt'}
    optimizer_param = 0.95   # momentum value for Momentum optimizer, or beta1 value for Adam

class OPTIMIZER:
    learning_rate =  {'min': 0.1, 'max': 0.9}
    nb_hiden_node1 = {'min': 10, 'max': 18}
    nb_hiden_node2 = {'min': 5, 'max': 14}
    batch_size     =     {'min': 10, 'max': 100}
    epochs = {'min': 500, 'max': 1000}
    C = {'min': 0.1, 'max': 0.9}
    #constrative_k = {'min': 1, 'max': 3}
    optimizer = ['skopt']   # ['bayesian', 'skopt']
    optimizer_param = {'min': 0.5, 'max': 0.99}


DATASET = Dataset()
#NETWORK = Network()
TRAINING = Training()
HYPERPARAMS = HYPERPARAMS ()
OPTIMIZER = OPTIMIZER ()
