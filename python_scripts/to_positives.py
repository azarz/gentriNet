"""
This small script extacts the positives from the classification results
"""


import numpy as np

def res2pos():
    """
    Simply extracts the positives when the results are within 1 file and saves them as a text file.
    """
    res = np.loadtxt('classification_results.dat','str')
    
    lst = []
    
    for row in res:
    	if row[3] == '1':
    		lst.append(row)
    
    lst = np.array(lst)
    
    np.savetxt('posititves.txt',lst,fmt='%s',delimiter=',')

#----------------------------------------------------------------------------------------------------

def res2pos_multi():
    """
    Extracts the positives of the gentriMap_allImages.py script 64 output files
    """
    for i in range(64):
    
        res = np.loadtxt('Results_1st_method/7th_try/classification_results_all_images_%i.dat'%i,'str')
    
        lst = []
    
        for row in res:
        	if row[3] == '1':
        		lst.append(row)
    
        lst = np.array(lst)
    
        np.savetxt('Results_1st_method/posititves_%i.txt'%i,lst,fmt='%s',delimiter=',')


#----------------------------------------------------------------------------------------------------

def multipos2singlepos():
    """
    Fuses the results of the previous funtion into a single file
    """
    lst = []
    for i in range(64):
    
        res = np.loadtxt('Results_1st_method/posititves_%i.txt'%i,'str',ndmin=1)
        lst.append(res)
        
    results = np.concatenate(lst)
    np.savetxt('Results_1st_method/posititves_0-63.txt',results,fmt='%s',delimiter=',')
