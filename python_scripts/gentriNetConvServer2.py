# -*- coding: utf-8 -*-
"""
This script was used to train iterations of CNNs to have a good classifications
of couples of images
"""

import numpy as np
from scipy import misc

# Importing the keras framework
import keras
from keras.models import Model
from keras.layers import Dense, Input, Flatten
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from keras.optimizers import SGD

# Svm_tools is a small script containing utilities for classification
import svm_tools

import random

# Importing utils from the sklearn library
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix

import tensorflow as tf

# This line forces the usage of GPU #2
with tf.device('/gpu:0'):
    
    # fix random seed for reproducibility
    #np.random.seed(7)
    
    # Forces the keras images fomat to have the number of channels last
    K.set_image_dim_ordering('tf')
    
    # Variable corresponding to the images size in pixels. 224 is the usual value
    IMG_SIZE = 224
    
    def compute_accuracy(predictions, labels):
        '''Function that determines the score to compute. Here it is the cohen kappa
        score.
        '''
        return cohen_kappa_score(predictions,labels)
    
    # Definition of the network architecture using keras functional API 
    # (useful for complex models) (https://keras.io/getting-started/functional-api-guide/)    
    input_img = Input(shape=(IMG_SIZE,IMG_SIZE,3))
    
    # VGGNet-19 Architecture (identical to https://github.com/fchollet/keras/blob/master/keras/applications/vgg19.py)
    x = ZeroPadding2D((1, 1), input_shape=(IMG_SIZE, IMG_SIZE,3))(input_img)
    x = Conv2D(64, 3, activation='relu')(x)
    x = ZeroPadding2D()(x)
    x = Conv2D(64, 3, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    
    x = ZeroPadding2D()(x)
    x = Conv2D(128, 3, activation='relu')(x)
    x = ZeroPadding2D()(x)
    x = Conv2D(128, 3, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    
    x = ZeroPadding2D()(x)
    x = Conv2D(256, 3, activation='relu')(x)
    x = ZeroPadding2D()(x)
    x = Conv2D(256, 3, activation='relu')(x)
    x = ZeroPadding2D()(x)
    x = Conv2D(256, 3, activation='relu')(x)
    x = ZeroPadding2D()(x)
    x = Conv2D(256, 3, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    
    x = ZeroPadding2D()(x)
    x = Conv2D(512, 3, activation='relu')(x)
    x = ZeroPadding2D()(x)
    x = Conv2D(512, 3, activation='relu')(x)
    x = ZeroPadding2D()(x)
    x = Conv2D(512, 3, activation='relu')(x)
    x = ZeroPadding2D()(x)
    x = Conv2D(512, 3, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    
    x = ZeroPadding2D()(x)
    x = Conv2D(512, 3, activation='relu')(x)
    x = ZeroPadding2D()(x)
    x = Conv2D(512, 3, activation='relu')(x)
    x = ZeroPadding2D()(x)
    x = Conv2D(512, 3, activation='relu')(x)
    x = ZeroPadding2D()(x)
    x = Conv2D(512, 3, activation='relu')(x)
    out = MaxPooling2D(pool_size=(2,2))(x)
    
    # The vision model corresponds to the 'bottom' of the network
    # It corresponds to a feature extractor using convolutions   
    vision_model = Model(input_img, out)

    # It is advised to initialize the weights of the feature extractor
    # with pre-trained models. In this example it is the vgg19 weights
    # of the first siamese model I trained 
#    weights_path = 'vgg19_branch.h5'
#    
#    vision_model.load_weights(weights_path)

    # In order to prevent overfitting, as advised in the keras documentation,
    # we freeze the 18 first convolutional layers (corresponding to the first 2 blocks)	
    for layer in vision_model.layers[:18]:
        layer.trainable = False
    
    # Definition of the 2 inputs
    img_a = Input(shape=(IMG_SIZE,IMG_SIZE,3))
    img_b = Input(shape=(IMG_SIZE,IMG_SIZE,3))

    # Outputs of the vision model corresponding to the inputs
    # Note that this method applies the 'tied' weights between the branches    
    out_a = vision_model(img_a)
    out_b = vision_model(img_b)

    # Concatenation of these ouputs    
    concat = keras.layers.concatenate([out_a, out_b])
    
    # Definition of the top layers    
    concat = Flatten()(concat)
    concat = Dense(4096, activation='relu',kernel_initializer='glorot_normal')(concat)
    concat = Dense(4096, activation='relu',kernel_initializer='glorot_normal')(concat)
    concat = Dense(1000, activation='relu',kernel_initializer='glorot_normal')(concat)
    main_out = Dense(1, activation='sigmoid')(concat)
    
    # The classification model is the full model: it takes 2 images as input and
    # returns a number between 0 and 1. The closest the number is to 1, the more confident
    # it is that there was a change   
    classification_model = Model([img_a, img_b], main_out)

    # Alternatively to loading weights at the branch level, we can load the weights of the 
    # classification model if we have already trained one.    
    weights_path = 'vgg19_siamese_8th_try.h5'
    
    classification_model.load_weights(weights_path)

    # Definition of the SGD optimizer, that allows to fine tune learning rate (very important)    	
    sgd = SGD(lr=1e-6, decay=1e-6, momentum=0.5, nesterov=True)
    # Compilation of the model using a loss used for binary problems    
    classification_model.compile(loss='binary_crossentropy', optimizer=sgd)
    
    
    # Array that contains all the results of the training dataset    
    all_results = []
    all_results = np.loadtxt('retrain.txt',str)
                
    num_res = len(all_results)
    
    # Array that contains the kappa score of each iteration
    kappas = []
    # number of wanted iterations. Each iteration takes a different set of negatives so
    # keep this number high (depending on the learning rate) when training the model
    n_iter = 150
    # Initialisation of a confusion matrix
    conf_mat = np.zeros((2,2))
    
    for iteration in range(n_iter):
        print(iteration/n_iter)
        # Each iteration takes a different set of results
        results = []
        
        # We append all the positive cases to the results
        for row in all_results:
            if row[-1]=='1':
                results.append(row)
        
        # We count the number of positives        
        num_yes = len(results)
        
        # We append 5 times more negatives than positives
        while len(results) < 6*num_yes:
            index = random.randint(0, num_res-1)
            row = all_results[index]
            if row[-1]=='0':
                results.append(row)
        
        # X is the array of image couples, y the array of corresponding labels
        X = []
        y = []
        
        for row in results:
            # The label is the last value of the row
            y.append(int(row[-1]))
            
            path1 = row[0]
            path2 = row[1]
            
            # We read the images and resize them
            img1 = misc.imread(path1)
            img2 = misc.imread(path2)
            img1 = misc.imresize(img1, (IMG_SIZE,IMG_SIZE))
            img2 = misc.imresize(img2, (IMG_SIZE,IMG_SIZE))
            X.append([img1,img2])
        
        X = np.array(X)
        y = np.array(y)
        # We use the utility prune_dataset2 to have 1000 negatives and 200 positives
        X, y = svm_tools.prune_dataset2(X, y, [1000,200])
        
        # Zero-centering the data
        X = X.astype(float) - np.mean(X, axis=0)
        
        # Splitting the data so the test is 33.33%, validation 11.11% and training 55.56% of the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333333)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.16666667)
        
        # Fitting the model.
        classification_model.fit([X_train[:,0], X_train[:,1]], y_train, batch_size=60, 
                                 epochs=10, validation_data=([X_val[:,0], X_val[:,1]], y_val))
        
        # Predictions to qulify the classification quality
        pred = (classification_model.predict([X_train[:, 0], X_train[:, 1]]) > 0.5).astype(int)
        tr_acc = compute_accuracy(pred, y_train)
        pred = (classification_model.predict([X_test[:, 0], X_test[:, 1]]) > 0.5).astype(int)
        te_acc = compute_accuracy(pred, y_test)
        kappas.append(te_acc)
        conf_mat = conf_mat + confusion_matrix(y_test, pred)		
        
        #print('* Kappa on training set: %f' % (tr_acc))
        #print('* Kappa on test set: %f' % (te_acc))
		
		
kappas = np.array(kappas)
print('')
print("Kappa: %0.10f (+/- %0.2f)"  % (kappas.mean(), kappas.std()))
print('')
print(conf_mat)

# These first 3 lines save the model as a json file
#model_json = classification_model.to_json()
#with open('vgg19_siamese_4th_ver.json', 'w') as jsonfile:
#	jsonfile.write(model_json)

# This last line is very important: it saves the weights of the model you trained for
# so long (warning: the weights files live up to their name and are very big, up to 1GB with this model)
# Warning: don't forget to change the name at each iteration
classification_model.save_weights('vgg19_siamese_10th_try.h5')

