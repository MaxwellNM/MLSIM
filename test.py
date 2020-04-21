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
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
#np.random.seed(1337)  # for reproducibility

LS="SG"
data =  np.load(DATASET.path_data2[LS]["data"])
target = np.load(DATASET.path_data2[LS]["label"])
if (np.shape(target)!=2):
    target = np.expand_dims(target,1)
# Split into training, validation, and test sets
if TRAINING.test_percent is None:
    train, traint, valid, validt, test, testt = split_data(data, target,TRAINING.test_percent)
else:
    X_train, X_test, y_train, y_test = split_data(data, target,TRAINING.test_percent)
print("shape training ",X_train.shape)

model = DecisionTreeClassifier(criterion="entropy", max_depth=10) #SVC(C=10,probability=False)#LogisticRegression ()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
print ("Model accuracy :",accuracy_score(y_test,predictions))
print ("Confusion matrix :\n",confusion_matrix(y_test,predictions))