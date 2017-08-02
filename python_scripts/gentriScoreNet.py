# -*- coding: utf-8 -*-
"""
This script (untested) aims to train a convolutional neural network to rate street view imagery
about how wealthy it looks
"""

import numpy as np
from scipy import misc

# Importing the neural net framework
from keras.models import Model
from keras.layers import Dense, Input, Flatten
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from keras.optimizers import SGD

from keras.models import model_from_json

# Useful library to read the training csv file
import csv

# Important library when wanting to find file and folder paths
import glob

# importing sklearn utilities to slit the data and calculate the r2 score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import trueskill

import tensorflow as tf

# This line forces the usage of CPU for machines with low memory GPUs
# Replace cpu:0 by gpu:0 or whatever index of the gpu you're using on other machines
with tf.device('/cpu:0'):
    
    # fix random seed for reproducibility
    #np.random.seed(7)
    
    # Forces the keras images fomat to have the number of channels last
    K.set_image_dim_ordering('tf')
    
    # Variable corresponding to the images size in pixels. 224 is the usual value
    IMG_SIZE = 224
    
    def compute_accuracy(predictions, labels):
        '''Function that determines the score to compute. Here it is the R² score since
        it is a regression problem
        '''
        return r2_score(predictions,labels)
    
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
    # of the convolutional layers trained on imageNet (https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5)
    weights_path = 'vgg19_weights.h5'
    
    vision_model.load_weights(weights_path)
    
    # Definition of the top layers
    concat = Flatten()(out)
    concat = Dense(4096, activation='relu',kernel_initializer='glorot_normal')(concat)
    concat = Dense(4096, activation='relu',kernel_initializer='glorot_normal')(concat)
    concat = Dense(1000, activation='relu',kernel_initializer='glorot_normal')(concat)
    main_out = Dense(1, activation='sigmoid')(concat)
     
    # The classification model is the full model: it takes an image as input and
    # returns an estimated rating
    classification_model = Model(input_img, main_out)
    
    # Definition of the SGD optimizer, that allows to fine tune learning rate (very important)
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
    # Compilation of the model using a loss used for rating models
    classification_model.compile(loss='mean_squared_error', optimizer=sgd)

    # Initialisation of the arrays containing the results and the scores
    results = []
    # Dictionary of the form {image_id: trueskill.Rating}
    image_scores = {}
    
    # Environment variables for trueskill
    # Beta is initialized as in this paper https://arxiv.org/pdf/1608.01769v2.pdf
    # The draw probability was estimated with the training data (number of draws/total)
    trueskill.BETA = 25/3
    trueskill.DRAW_PROBABILITY = 0.18
    
    # Reading the training data file
    with open("web_interface/votes.csv") as votefile:
        reader = csv.reader(votefile)      
        for row in reader:
            # We ignore the first row
            if row[0] != 'id1':
                results.append(row)
                
                # If an image in the current row has no score yet, it is initialized
                if row[0] not in list(image_scores):
                    image_scores[row[0]] = trueskill.Rating()
                if row[1] not in list(image_scores):
                    image_scores[row[1]] = trueskill.Rating()
                       
    # Going through the results
    for row in results:
        # We update the rankings according to the documentation (http://trueskill.org/)
        if row[4] == 'left':
            image_scores[row[0]], image_scores[row[1]] = trueskill.rate_1vs1(image_scores[row[0]], image_scores[row[1]])                    
        elif row[4] == 'right':
            image_scores[row[1]], image_scores[row[0]] = trueskill.rate_1vs1(image_scores[row[1]], image_scores[row[0]])
        else:
            image_scores[row[1]], image_scores[row[0]] = trueskill.rate_1vs1(image_scores[row[1]], image_scores[row[0]], drawn=True)
    
    scores = []

    # The rank of an image is in fact the mu value of the Rating object
    for key in list(image_scores):
        scores.append(image_scores[key].mu)
        
    # Fetching the min and max in order to rescale the scores
    mini, maxi = min(scores), max(scores)
    
    # Rescaling the scores between 0 and 1
    coef = 1/(maxi - mini)
    
    for key in list(image_scores):
        image_scores[key] = coef * (image_scores[key].mu - maxi) + 1
                    
    # Fetching all the images paths
    images = glob.glob('voteimages/*.jpg')

    # X is the array of images, y the array of corresponding labels
    X = []
    y = []
    
    # Initializing the data for the clasification
    for key in list(image_scores):
        # The lqbel is the score of the image
        y.append(image_scores[key])
        
        image_index = key
        image_index = int(image_index)
        
        # The path of the image is the value in the 'images' array corresponding to the key in the
        # image_scores dictionary
        path = images[image_index]
        
        # Reading and resizing the image (bilinear interpolation by default)
        img = misc.imread(path)
        img = misc.imresize(img,(IMG_SIZE,IMG_SIZE))
    
        X.append(img)
    
    # Convertin the arrays to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Zero-centering the data
    X = X.astype(float) - np.mean(X, axis=0)
    
    # Splitting the data so the test is 70%, validation 5% and training 25% of the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.7, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.16666667)
    
    # Fitting the model. I think you should fine-tune the batch_size and number of epochs
    classification_model.fit(X_train, y_train, batch_size=100, 
                                 epochs=100, validation_data=(X_val, y_val))
    
    # Trying a predicition to calculate the R squared value
    pred = classification_model.predict([X_test[:, 0], X_test[:, 1]])
    te_acc = compute_accuracy(pred, y_test)
    print(te_acc)


# If the classification results are satisfying, uncomment the following lines:

## These first 3 lines save the model as a json file
#model_json = classification_model.to_json()
#with open('vgg19_ranking.json', 'w') as jsonfile:
#	jsonfile.write(model_json)

## This last line is very important: it saves the weights of the model you trained for
## so long (warning: the weights files live up to their name and are very big, up to 1GB with this model)
#classification_model.save_weights('vgg19_ranking_weights.h5')