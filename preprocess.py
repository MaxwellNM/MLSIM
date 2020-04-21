# -*- coding: utf-8 -*-

import os
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.model_selection import train_test_split

#xls_path = './data/mdl_logstore_standard_log.csv'
def load_excel_data(xls_path):
    fname = xls_path.split(os.sep)[-1]
    t_ext = fname.split(".")[-1]
    if t_ext == "csv":
        gl_moodle = pd.read_csv(xls_path)
    else:
        gl_moodle = pd.read_excel(xls_path)
    return gl_moodle

def get_numpy_data(gl_moodle):
    numpy_data = gl_moodle.values
    return numpy_data

"""
@Input: lstyle: list of  learning styles categories
@Input: ls_student: list of student involved
@Input: gl: log file of student activities in the vle
@Output: gamma: matrix of learning traces for each student
"""
def get_features_matrix_learninstyle(gamma,gl,lstyle,ls_student):
    """Take in parameters a list of features , list of id, log mtrix returns the feature matrix of the learning style"""
    #gamma = np.zeros((len(ls_student),len(lstyle)),dtype=np.int8)
    i=0
    for s in ls_student:
        lis=[]
        for lst in lstyle:
            s_gl = gl[(gl.username==s) & (gl.eventname==lst)]#take sub set of event occurence in the log
            lis.append(len(s_gl))
        gamma[i,:] = lis
        i+=1
        if i%10==0:
            print ('processed {}/{}'.format(i,len(ls_student)) )
    return gamma

def load_data(path_data, ext="txt"):
    if os.path.isfile(path_data):
        print ("File exist")
    else:
        print ("File not exist")
    if ext == "txt":
        mat = np.loadtxt(path_data,dtype=np.int32)
    else:
        mat = np.load(path_data,dtype=np.int32)
    return mat

# Impute missing data
"""
@Input: mat_abs: matrix of extracted and undigitized learning traces
@Input: index_relevant_features index list of students' most prefered learning activities regarding the learning style LS_XX
@Output: imp_mat : name list of students' most prefered learning activities regarding the learning style LS_XX
"""
def impute_missed_features(index_relevant_features, mat_abs):
    pmod_data =mat_abs[:,index_relevant_features]
    imputer = Imputer(missing_values=0, strategy='median', axis=0)
    imp_mat =imputer.fit_transform(pmod_data)
    return imp_mat

#Saved the preprocessed data
""" Save the extracted and Imputated data in the excel format.
@Input: imp_mat: matrix of extracted and Imputed missing values of  learning traces
@Input: target : list of learning style id
@Input: feature_names : list of learning activities' (features) names
@Input: path_save : path to save the matrix in xls file 
@Output: mat_xls : pandas matrix for notebook worksheet
"""
def save_features_xls(imp_mat, target,feature_names, path_save, mat_xls):
    mat_xls = pd.DataFrame(imp_mat, columns=feature_names)
    mat_xls['target'] = target
    mat_xls.to_excel(path_save)


def split_data(data,target,test_percent=None):
    # Randomly order the data
    order = list(range(np.shape(data)[0]))
    np.random.shuffle(order)
    data = data[order,:]
    target = target[order,:]
    if test_percent is None: # 50% train, 25% valid, 25% test
        train = data[::2,:]
        train_target = target[::2]
        valid = data[1::4,:]
        valid_target = target[1::4]
        test = data[3::4,:]
        test_target = target[3::4]
        return train, train_target, valid, valid_target, test, test_target
    else:
        X_train, X_test, Y_train, Y_test  = train_test_split(data,target, test_size=test_percent,random_state=0)
        return X_train, X_test, Y_train, Y_test 


    