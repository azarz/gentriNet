# -*- coding: utf-8 -*-
"""
This is the actual script I used to download the ottawa_image_db
"""

import numpy as np
from scipy import misc

import gist
import streetview

from osgeo import ogr

import random

API_KEY = "<yourAPIkey>"

#------------------------------------------------------------------------------

def runOttawa(n_loc):
    """
    Downloads Google StreetView images of random points in Ottawa
    
    :param n_loc: number of locations to download. CAUTION: some points have no images, so it's not the exact number of subdirectories created
    """
    
    DIRECTORY = "../voteimages"
    
    ds = ogr.Open('C:/Users/msawada/Desktop/Urban_RAT/Urban_RAT_inventory.dbf')
    layer = ds.GetLayer()
    
    for i in range(n_loc):
        print('%.2f' %(i*100/n_loc) + " %")
        
        index = random.randint(0,len(layer))
        feature = layer[index]
        
        lon = feature.GetGeometryRef().GetX()
        lat = feature.GetGeometryRef().GetY()
        
        bearing_road = feature.bearing
        
        if bearing_road == None:
            heading = ''
        else:
            heading = bearing_road + 90*np.sign(random.random() - 0.5)
        
        folder = DIRECTORY
        
        panIds = streetview.panoids(lat, lon)
        
        if len(panIds) > 0:
    
            for pan in panIds:
                img = streetview.api_download(pan["panoid"], heading, folder, API_KEY, fov=80, pitch=0, year=pan["year"])
                if img != None:
                    full_size = misc.imread(img)
                    resized = misc.imresize(full_size, (64,64))
                    desc = gist.extract(resized)
                    np.savetxt(img + '.txt', desc)

#------------------------------------------------------------------------------
