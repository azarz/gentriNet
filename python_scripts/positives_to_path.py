# -*- coding: utf-8 -*-
"""
This small script converts a file of positive results from the classification
so the rows include the path to the file and not an ID
It also saves the couples so they can be visualized.
It also converts the results of a review of the results so they can be appended to the retrain.txt
used in the iterations
"""

import csv

import glob

import numpy as np

from scipy import misc

import matplotlib.pyplot as plt


def positives2couples(saveimages=False, review2retrain=False):
    """
    Converts the result of the to_positives.py script to rows with the file path on the computer
    Can also save the couples of images in the reviewcouples folder if saveimages is True
    Can also convert the train_review.txt (same as .csv but without the first row) from the 
    review interface to a file that can be appended to retrain.txt to train the model on another iteration
    with the parameter review2retrain as True
    """
    pos = []
    
    with open("posititves_0-63.txt") as posit:
        reader = csv.reader(posit)      
        for row in reader:
            pos.append(row)
                
    couples = []
    
    big_folder = 'D:/Amaury/Desktop/ottawa_image_db'
    
    # Parsing the file row to deduce the file name
    for row in pos:
        folder = str(row[0]) + ',' + str(row[1])
        y1 = str(row[2])
        y2 = str(row[3])
        
        if y1 == y2:
            im1 = glob.glob(big_folder+'/'+folder+'/'+y1+'*.jpg')[0]
            im2 = glob.glob(big_folder+'/'+folder+'/'+y2+'*.jpg')[1]
        else:
            im1 = glob.glob(big_folder+'/'+folder+'/'+y1+'*.jpg')[-1]
            im2 = glob.glob(big_folder+'/'+folder+'/'+y2+'*.jpg')[0]
            
        couples.append([im1,im2])
        
    couples = np.array(couples)
    
    np.savetxt('couples.txt',couples,fmt='%s',delimiter=',')
    
    # Saving the image couples using matplotlib.pyplot
    # They are saved in the reviewcouples folder. Don't forget to backup and have an empty
    # folder with that name
    if saveimages:
        plt.ioff()
        for rowind in range(len(couples)):
            row = couples[rowind]
            plt.figure(figsize=(10, 10), dpi=120)
            a = misc.imread(row[0])
            b = misc.imread(row[1])
            u = plt.subplot(211)
            plt.imshow(a)
            u = plt.subplot(212)
            plt.imshow(b)
            plt.savefig('reviewcouples/%i.jpg'%rowind)
            plt.close()

    # Converting the results of the review to data undertandable to the gentriNetCConvServer2.py script
    # Just copy the csv  form the web review interface, delete the header row and change the extension to .txt
    if review2retrain:
        train = np.loadtxt('train_review.txt', dtype=int,delimiter=',')
        
        tr = np.unique(train, axis=0)
        
        review_results = []
        
        for row in tr:
            couple = couples[row[0]]
            im1 = couple[0]
            im2 = couple[1]
            review_results.append([im1,im2,row[1]])
        np.savetxt('retrain_new.txt', review_results, fmt='%s')
