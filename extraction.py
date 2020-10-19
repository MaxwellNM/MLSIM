# -*- coding: utf-8 -*-

import os
import numpy as np
#from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd
from matplotlib import pyplot as plt 
import math
from sklearn.preprocessing import normalize
#from utils import *

"""
@Input: matrix of learning traces summarized
@Output: matrix of features in [0,1,2,3] : those are the hint of Jason Bernard (2017) & Sabine Graf (2009)
"""
def discretize_learning_traces(mat_abs):
    count_data = mat_abs.shape[0]*mat_abs.shape[1]
    nbbins = math.floor(np.sqrt(count_data))
    bins = [0,1,2,3]#range(nbbins)#[0,1,2,3]
    digitized = np.digitize(mat_abs, bins)
    bin_means = [mat_abs[digitized == i].mean() for i in range(1, len(bins))]
    #print("bins ",bins," digitized 2 ",digitized[0:2]," mat abs ",mat_abs[0:2])
    return digitized

def discretize_learning_traces2(mat_abs):
    est = KBinsDiscretizer(n_bins=9, encode='ordinal', strategy='kmeans')
    est.fit(mat_abs)
    digitized = est.transform(mat_abs)
    return digitized

def print_box_blot(data,msg):
    fig1, ax1 = plt.subplots()
    ax1.set_title('Box Plot '+msg)
    ax1.boxplot(normalize(data))
    plt.show()

def discretize_learning_traces3(mat_abs):
    count_data = mat_abs.shape[0]*mat_abs.shape[1]
    nbbins = math.floor(np.sqrt(count_data))
    bins = [0,1,2,3]#range(nbbins)#[0,1,2,3]
    digitized = np.digitize(mat_abs, bins)
    bin_means = [mat_abs[digitized == i].mean() for i in range(1, len(bins))]
    #print("bins ",bins," digitized 2 ",digitized[0:2]," mat abs ",mat_abs[0:2])
    return digitized

"""We used here a litterature approach to have a prior estimation of the learning style (building labels) """
"""S.Graf,2010 & J. Bernard, 2017 """
"""
@Input: matrix of learning traces digitized
@Output: y_ls_AR in 0 and 1  it tells if yes ornot the student belong to the specific learning style dimension
@Output: liked_feat_student list of learning activities most prefered by students'
@Output: LS_percent : percentage of student learning style similarity 
"""

def build_target(digit_matrix,mat_abs):
    LS = np.sum(digit_matrix,axis=1)/digit_matrix.shape[1]
    max_val =np.max(LS)

    #student preference learning activity
    #value of learning activity not appreciate by student 
    feature_min_student = np.argmin(mat_abs,axis=1)
    #value of learning activity more appreciate by student 
    feature_max_student = np.max(mat_abs,axis=1)

    #value of learning activity or component more appreciate by student
    liked_feat_student = np.argmax(mat_abs,axis=1)

    LS_percent =LS*100/max_val #graf compute for the theoritical label of the learning style (computing theoritical learning style)

    y_ls_AR_percent = np.vstack(LS_percent)
    y_ls_AR = np.where(y_ls_AR_percent<=50.0,0,1)

    print('unliked E-learning component or activity ',feature_min_student)

    return y_ls_AR, liked_feat_student,LS_percent

"""We extract the feature prefered mostly by each student
@Input: LS_XX: list of  learning styles categories
@Input: liked_feat_student list of learning activities most prefered by students'
@Output: index_relevant_features index list of students' most prefered learning activities regarding the learning style LS_XX
@Output: Desc : name list of students' most prefered learning activities regarding the learning style LS_XX
"""
#Define a function to print relevant features.
def extract_features_learning(LS_XX, liked_feat_student):
    index_relevant_features = []
    l_behav = []
    for elt in liked_feat_student:
        if not elt in index_relevant_features:
            index_relevant_features.append(elt)
            l_behav.append(LS_XX[elt]) 
    Desc = np.vstack(l_behav)
    return index_relevant_features,Desc

#binarized data
""" transform features in binary format to train in the DBN.
@Input: digitized: matrix of extracted and digitized learning traces
#@Input: l_behav : list of learning activities id
@Input: number_feature_behav : list of learning activities id
@Output: xdigitize : binarized data to train by te model
"""
#def binarized_features2(digitized, number_feature_behav):
#    print("number features: ",number_feature_behav)
#    onehotencoder = OneHotEncoder(categorical_features= range(0,number_feature_behav))
#    xdigitize = onehotencoder.fit_transform(digitized).toarray()
#    return xdigitize

def binarized_features(data,nb_features):
    enc = {}
    enc[1] = [0, 0, 1]
    enc[2] = [0, 1, 0]
    enc[3] = [0, 1, 1]
    enc[4] = [1, 0, 0]
    xp = np.zeros((data.shape[0],nb_features))
    for i,row in enumerate(data):
        l = [enc[elt] for elt in row]
        xp[i] = np.hstack(l)
    return xp

def plot_histogram(matrix,histname="histogram",n=10):
    #count_data = matrix.shape[0]*matrix.shape[1]
    #nbr_bins = math.floor(np.sqrt(count_data))
    hist, bins = np.histogram(matrix,bins=range(0,n))
    plt.hist(hist, bins) 
    plt.title(histname) 
    plt.show()