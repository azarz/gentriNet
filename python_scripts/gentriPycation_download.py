# -*- coding: utf-8 -*-
"""
Script used to download images in a folder with this structure: mainfolder/latlon/images.jpg
(so each location has its folder) from a set of locations in a shapefile.
It also processes the GIST vector of each image (at different scales) and possibly the 
dense SIFT, and saves them as text files in the same directory as the images.
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

# Please use your own Google API key and not the default one
API_KEY = "AIzaSyBIgKy3bpsMnaKGuUexZPP9I9C7xjA8jX0"

#------------------------------------------------------------------------------
def sift(*args, **kwargs):
    """
    Wrapper for the OpenCV SIFT function.
    """
    try:
        return cv2.xfeatures2d.SIFT_create(*args, **kwargs)
    except:
        return cv2.SIFT()

def dsift(img, step=5):
    """
    Implementation of the dense SIFT. The SIFT indicator is calculated each
    'step' pixels of the image.
    """
    keypoints = [
        cv2.KeyPoint(x, y, step)
        for y in range(0, img.shape[0], step)
        for x in range(0, img.shape[1], step)
    ]
    features = sift().compute(img, keypoints)[1]
    features /= features.sum(axis=1).reshape(-1, 1)
    return features


#------------------------------------------------------------------------------

def runOttawa(n_loc):
    """
    Downloads Google StreetView images of random points in the shapefile, those points
    have weighted probabilities to be chosen in their attributes.
    
    :param n_loc: number of locations to download. CAUTION: some points have no images, so it's not the exact number of subdirectories created
    """
    DIRECTORY = "trainottawa"
    
    # Opening the layer and fetching the weights, and the normalizing them
    ds = ogr.Open('ottawashp/ottawa_4326_clipped_weighted_points.shp')
    layer = ds.GetLayer()
    weights = np.array([feature.weight for feature in layer])
    weights_sum = np.sum(weights)
    weights = weights / weights_sum
    
    indices = np.array(range(len(layer)))
    
    for i in range(n_loc):
        print('%.2f' %(i*100/n_loc) + " %")
        
        # We select randomly the index according to the weights
        index = int(choice(indices, p=weights))
        feature = layer[index]
        
        # We fetch the coordinates
        lon = feature.GetGeometryRef().GetX()
        lat = feature.GetGeometryRef().GetY()
        
        # We fetch the bearing of the road
        bearing_road = feature.bearing
        
        # If there is non, we let the heading at its default value
        if bearing_road == None:
            heading = ''
        # Else, we take randomly a heading of +90 or -90 degrees, so the image faces a building
        else:
            heading = bearing_road + 90*np.sign(random.random() - 0.5)
        
        # Creating the folder name
        folder = DIRECTORY + '/new-%2.6f,%2.6f' %(lat,lon)
        oldfolder = DIRECTORY + '/%2.6f,%2.6f' %(lat,lon)
        
        # using the streetview module to fetch the panoids corresponding to the position
        panIds = streetview.panoids(lat, lon)
        
        if len(panIds) > 0:
            # We only proceed if the folder doesn't already exist
            if not os.path.exists(oldfolder):
                if not os.path.exists(folder):
                    os.makedirs(folder)
                    
                    # We download each image
                    for pan in panIds:
                        img = streetview.api_download(pan["panoid"], heading, folder, 
                                                      API_KEY, fov=90, pitch=10, year=pan["year"])
                        if img != None:
                            # We process and the GIST vector at different scales, and save them
                            # as text files in the same folder
                            full_size = misc.imread(img)
                            resized = misc.imresize(full_size, (256,256))
                            desc = gist.extract(resized)
                            np.savetxt(img + '256.txt', desc)
                            resized = misc.imresize(full_size, (128,128))
                            desc = gist.extract(resized)
                            np.savetxt(img + '128.txt', desc)
                            resized = misc.imresize(full_size, (64,64))
                            desc = gist.extract(resized)
                            np.savetxt(img + '64.txt', desc)
                            resized = misc.imresize(full_size, (32,32))
                            desc = gist.extract(resized)
                            np.savetxt(img + '32.txt', desc)
                            # Processing and saving the dense sift vector
                            sift_desc = dsift(misc.imresize(img, (64,64)))
                            np.savetxt(img + '.nfo', sift_desc)