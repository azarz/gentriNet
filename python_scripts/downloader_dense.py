# -*- coding: utf-8 -*-
"""
Created on Thu May 25 09:11:23 2017

@author: amaury
"""

import numpy as np
from scipy import misc

import gist
import streetview

from osgeo import ogr

import os

API_KEY = "YOURAPIKEY"

#------------------------------------------------------------------------------

def runOttawa():
    """
    Downloads Google StreetView images of random points in Ottawa
    
    :param n_loc: number of locations to download. CAUTION: some points have no images, so it's not the exact number of subdirectories created
    """
    
    DIRECTORY = "D:\Amaury\Desktop\ottawa_image_db"
    
    ds = ogr.Open('C:/Users/msawada/Desktop/Urban_RAT/Urban_RAT_inventory_4326.dbf')
    layer = ds.GetLayer()
    
    n_loc = len(layer)
    #done: range(3000)
    for i in range(3000,6000):
        print('%.2f' %(i*100/n_loc) + " %")
        
        index = i
        feature = layer[index]
        
        lon = feature.GetGeometryRef().GetX()
        lat = feature.GetGeometryRef().GetY()
        
        folder = DIRECTORY + '/%2.6f,%2.6f' %(lat,lon)
        
        panIds = streetview.panoids(lat, lon)
        
        if len(panIds) > 0:
            if not os.path.exists(folder):
                os.makedirs(folder)
                for pan in panIds:
                    img = streetview.api_download(pan["panoid"], folder, API_KEY, fov=80, pitch=0, year=pan["year"],month=pan["month"])
                    if img != None:
                        full_size = misc.imread(img)
                        resized = misc.imresize(full_size, (64,64))
                        desc = gist.extract(resized)
                        np.savetxt(img + '64.txt', desc)

#------------------------------------------------------------------------------
