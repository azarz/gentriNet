# -*- coding: utf-8 -*-
"""
This script is used to classify all the images of the database dwnloaded all over Ottawa
"""

import numpy as np
from scipy import misc

from keras.optimizers import SGD
from keras.models import model_from_json

import glob

import tensorflow as tf

# Variable corresponding to the images size in pixels. 224 is the usual value
IMG_SIZE = 224

# This line forces the usage of GPU #2
with tf.device('/gpu:1'):
    
    # We load the structure of the classification model, stored as a JSON file
    model_file = open('vgg19_siamese_4th_ver.json', 'r')
    loaded_model = model_file.read()
    
    classification_model = model_from_json(loaded_model)
    
    # We also load the weights we previously trained
    classification_model.load_weights('vgg19_siamese_7th_try.h5')
    
    # The optimizer isn't very important at this point since the model is already trained
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
    # We use the binary crossentropy loss as our classification problem is binary
    classification_model.compile(loss='binary_crossentropy', optimizer=sgd)
    
    # We fetch all the folder paths in the folder containing the images in an array
    locations = glob.glob('D:/Amaury/Desktop/ottawa_image_db/*')
    
    num_locations = len(locations)
	
    print('loading locations')
	
    # We proceed the classification by steps of 1/64 of the dataset so we don't run out of memory
    for portion in range(64):
        print(portion/64)
		
        # X is the array containing the data to classify
        X = []
        # Array of the corresponding location and year
        loc_year = []
		
        # We calculate the indices bounds of the current portion
        lower_bound = int(np.ceil(num_locations * portion/64))
        upper_bound = int(np.ceil(num_locations * (portion+1)/64))
		
        for loc_i in range(lower_bound, upper_bound):
			
            #print((loc_i-lower_bound)/(upper_bound-lower_bound))
			
            # we fetch the folder path of the current location
            loc = locations[loc_i]
			
            # We store all the image paths of this folder in an array
            images = glob.glob(loc + '/*.jpg')
			
            # If there is more the 1 image at this location
            if len(images) > 1:
                # latlon is the part of the name of the file that corresponds to the position
                # i.e. the folder name
                latlon = loc[-20:]
				
                index = 0
				
                # We go through the folder and add each couple to the dataset
                while index < (len(images) - 1):
					
                    path1 = images[index]
                    path2 = images[index + 1]
					
                    # The year corresponds with the first 4 characters in the file name
                    year1 = path1.split('\\')[-1][:4]
                    year2 = path2.split('\\')[-1][:4]

                    img1 = misc.imread(path1)
                    img2 = misc.imread(path2)
                    img1 = misc.imresize(img1, (IMG_SIZE,IMG_SIZE))
                    img2 = misc.imresize(img2, (IMG_SIZE,IMG_SIZE))
				
                    X.append([img1,img2])
                    loc_year.append([latlon,year1,year2])
					
                    index += 1
		
        X = np.array(X)
		
        # Zero-centering
        mean = np.mean(X, axis=0)
        X = X.astype(float) - mean
	
        # Predicting the data class: if the confindence that there is a change is more than 50% (0.5)
        # We assume that ther is a change, and in other cases we assume there is not
        pred = (classification_model.predict([X[:, 0], X[:, 1]]) > 0.5).astype(int)
        loc_year = np.array(loc_year)
        
        # Saving the results of the portion as a text file
        concatenation = np.concatenate((loc_year, pred), axis=1)
        np.savetxt('classification_results_all_images_%i.dat'%portion, concatenation, fmt='%s')
