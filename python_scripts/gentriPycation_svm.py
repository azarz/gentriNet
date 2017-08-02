# -*- coding: utf-8 -*-
"""
This script uses SVM and GIST vectors (andpossibli dense SIFT) to classify pictures
"""

import numpy as np
from numpy.random import choice
from scipy import misc
import os

import gist
import streetview
import cv2

from osgeo import ogr

import random

import csv
import glob

from scipy.spatial.distance import euclidean

#https://pypi.python.org/pypi/feature-aggregation
from feature_aggregation import BagOfWords, Vlad, FisherVectors, LLC

from sklearn import svm
import svm_tools
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix


# Array that contains all the results of the training dataset
all_results = []

with open("web_interface_2/train.csv") as votefile:
    reader = csv.reader(votefile)      
    for row in reader:
        # We ignore the first row and the ones that were labeled as 'null'
        if row[5]!='null' and row[0] != 'id':
            all_results.append(row)
            
# We count the number of results
num_res = len(all_results)

# Array that contains the f1 score of each iteration
f1_scores = []
# Array that contains the kappa score of each iteration
kappas = []
# number of wanted iterations. Each iteration takes a different set of negatives
n_iter = 3
# Initialisation of a confusion matrix
conf_mat = np.zeros((2,2))

for iteration in range(n_iter):
    print(iteration/n_iter)
    # Each iteration takes a different set of results
    results = []
    
    # We append all the positive cases to the results
    for row in all_results:
        if row[5]=='yes' or row[5]==1:
            row[5]=1
            results.append(row)
    
    # We count the number of positives
    num_yes = len(results)
    
    # We append as many negatives as positives
    while len(results) < 2*num_yes:
        index = random.randint(0, num_res-1)
        row = all_results[index]
        if row[5]=='no' or row[5]==0:
            row[5]=0
            results.append(row)
            
    # We fetch an array of all the folders paths        
    folders = glob.glob('D:/Amaury/Desktop/trainottawa/*')

    # X is the data array and y the array of corresponding labels        
    X = []
    y = []
    
    
    # Bag of Visual Words approach for dense SIFT vectors
    
#    bow = FisherVectors(200)
#
#    sifts = []
#    
#    for row in results:
#        folder_index = ''
#        image_index = ''
#        ind = 0
#        
#        while row[0][ind] != '-':
#            folder_index += row[0][ind]
#            ind+=1
#        ind+=1
#        while row[0][ind] != '-':
#            image_index += row[0][ind]
#            ind+=1
#        folder_index = int(folder_index)
#        image_index = int(image_index)
#        image2_index = image_index + 1
#        
#        path1s = glob.glob(folders[folder_index] + '/*.nfo')[image_index]
#        path2s = glob.glob(folders[folder_index] + '/*.nfo')[image2_index]
#        
#        sift_1 = np.loadtxt(path1s)
#        sift_2 = np.loadtxt(path2s)
#        
#        sifts.append(sift_1)
#        sifts.append(sift_2)
#        
#
#    bow.fit(sifts)
    
    
    for row in results:
        # The label is the last value of the row
        y.append(row[-1])
        
        folder_index = ''
        image_index = ''
        ind = 0
        
        # We can find in the id of the row the index of the folder corresponding
        # to the location, and the indices of the images in that folder            
        while row[0][ind] != '-':
            folder_index += row[0][ind]
            ind+=1
        ind+=1
        while row[0][ind] != '-':
            image_index += row[0][ind]
            ind+=1
            
        folder_index = int(folder_index)
        image_index = int(image_index)
        image2_index = image_index + 1
        
        # From the indices we load we deduce the corresponding descriptor files
        # the txt files are for the GIST descriptors and the NFO ones are dense SIFT
        path1g = glob.glob(folders[folder_index] + '/*64.txt')[image_index]
#        path1s = glob.glob(folders[folder_index] + '/*.nfo')[image_index]
        path2g = glob.glob(folders[folder_index] + '/*64.txt')[image2_index]
#        path2s = glob.glob(folders[folder_index] + '/*.nfo')[image2_index]
        
        gist_1 = np.loadtxt(path1g)
        gist_2 = np.loadtxt(path2g)
        
        # Using the dense SIFT is very long so I do not recommand
        
#        sift_1 = np.loadtxt(path1s)
#        sift_2 = np.loadtxt(path2s)

#        transformed = bow.transform([sift_1, sift_2])
#        sift_1 = transformed[0]
#        sift_2 = transformed[1]
        
#        desc_1 = np.concatenate((gist_1,sift_1))
#        desc_2 = np.concatenate((gist_2,sift_2))
        desc_1 = gist_1
        desc_2 = gist_2
    
        
#        X.append((desc_2 - desc_1))
        X.append(np.concatenate((desc_1,desc_2)))

    X = np.array(X)
    
    y = np.array(y)

#    centroid = np.mean(X, axis=0)
#    distances =[]
#    for row in X:
#        distances.append(euclidean(centroid, row))


#    C_test = [0.1, 1, 5, 10, 100, 1000]
#    C_opt = svm_tools.cv_SVM_linear(X, y, 4, C_test)
#    print(C_opt)
    
#    clf = Pipeline([("chi2", AdditiveChi2Sampler()), ("svm", LinearSVC(C=10))])
#    clf=RandomForestClassifier(max_depth=10, n_estimators=1000, max_features=200)

#    svm_tools.RFE(X,y,3,C_test)
    
    clf = svm.SVC(kernel="linear", C=5)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    clf.fit(X_train, y_train)
    scores = cross_val_score(clf, X, y, cv=3, scoring='f1')
    f1_scores.append(scores)
    y_pred = clf.predict(X_test)
    kappas.append(cohen_kappa_score(y_test,y_pred))
    conf_mat = conf_mat + confusion_matrix(y_test, y_pred)
#    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#    print(classification_report(y_test, clf.predict(X_test)))

f1_scores = np.array(f1_scores).flatten()
kappas = np.array(kappas)
print("F1-Score: %0.2f (+/- %0.2f)" % (f1_scores.mean(), f1_scores.std()))
print("Kappa: %0.10f (+/- %0.2f)"  % (kappas.mean(), kappas.std()))
print('')
print(conf_mat)
