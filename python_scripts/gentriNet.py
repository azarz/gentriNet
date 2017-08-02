# -*- coding: utf-8 -*-
"""
This script uses Dense Neural Network on gist vectors to classify couples of images
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

import numpy as np
import csv
import glob
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from keras.optimizers import Adam
from keras.constraints import maxnorm

import tensorflow as tf

# fix random seed for reproducibility
#np.random.seed(7)

# This line forces the usage of CPU for machines with low memory GPUs
# Replace cpu:0 by gpu:0 or whatever index of the gpu you're using on other machines
with tf.device('/cpu:0'):
    
    # create model
    def create_baseline(n1=60,n2=30,learn_rate=0.01,init_mode='glorot_normal',dropout_rate=0.4, weight_constraint=2):
        model = Sequential()
        model.add(Dense(n1, kernel_initializer=init_mode, input_dim=1920, activation='relu',kernel_constraint=maxnorm(weight_constraint)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(n2, activation='relu',kernel_initializer='glorot_normal'))
        model.add(Dense(1, activation='sigmoid'))
        # Compile model
        optimizer = Adam(lr=learn_rate)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

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
    
    # Array that contains the kappa score of each iteration
    kappas = []
    # number of wanted iterations. Each iteration takes a different set of negatives so
    # keep this number high (depending on the learning rate) when training the model
    n_iter = 25
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
            
             # From these indices we can deduce the path of the corresponding GIST vectors
            path1g = glob.glob(folders[folder_index] + '/*64.txt')[image_index]
            path2g = glob.glob(folders[folder_index] + '/*64.txt')[image2_index]
            
            # We load the GIST vectors
            gist_1 = np.loadtxt(path1g)
            gist_2 = np.loadtxt(path2g)
        
            X.append(np.concatenate((gist_1,gist_2)))
        
        X = np.array(X)
        y = np.array(y)
#        X, y = svm_tools.prune_dataset(X, y, 50)
    
        # Zero-centering
        X -= np.mean(X, axis=0)

        # Definition of parameters for a grid-search fine-tuning
#        n1s = [60,120,240,480,960]
#        n2s = [30,60,120,240,480]
#        epochs = [50,75,100,125]
#        batches = [5,10,25]
#        optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
#        learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
#        init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
#        weight_constraint = [1, 2, 3, 4, 5]
#        dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
        # The parameter grid depends on the parameters we want to fine tune
#        param_grid = dict(n1=n1s, n2=n2s)
#        param_grid = dict(epochs=epochs, batch_size=batches)
#        param_grid = dict(optimizer=optimizer)
#        param_grid = dict(learn_rate=learn_rate)
#        param_grid = dict(init_mode=init_mode)
#        param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)
    
        # Definition of the classifier from the model
        clf = KerasClassifier(build_fn=create_baseline, epochs=125, batch_size=25)
        
        # Grid-Search
#        grid = GridSearchCV(estimator=clf, param_grid=param_grid)
#        grid_result = grid.fit(X, y)
#        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#        means = grid_result.cv_results_['mean_test_score']
#        stds = grid_result.cv_results_['std_test_score']
#        params = grid_result.cv_results_['params']
#        for mean, stdev, param in zip(means, stds, params):
#            print("%f (%f) with: %r" % (mean, stdev, param))
        

        # Splitting the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    
        # Fitting the model
        clf.fit(X_train, y_train)
        
        # Calculating the kappa score
        y_pred = clf.predict(X_test)
        kappas.append(cohen_kappa_score(y_test,y_pred))
        conf_mat = conf_mat + confusion_matrix(y_test, y_pred)


kappas = np.array(kappas)
print('')
print("Kappa: %0.10f (+/- %0.2f)"  % (kappas.mean(), kappas.std()))
print('')
print(conf_mat)

