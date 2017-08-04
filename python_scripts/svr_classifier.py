# -*- coding: utf-8 -*-
"""
Used to classify the results of the second method using SVR
"""

import numpy as np
from scipy import misc
import svm_tools

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

import csv
import glob

import trueskill

from sklearn import svm
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt



#"""
results = []
image_scores = {}

trueskill.BETA = 25/3
trueskill.DRAW_PROBABILITY = 0.18

with open("../web_interface/votes.csv") as votefile:
    reader = csv.reader(votefile)      
    for row in reader:
        if row[0] != 'id1':
            results.append(row)
            
            if row[0] not in list(image_scores):
                image_scores[row[0]] = trueskill.Rating()
            if row[1] not in list(image_scores):
                image_scores[row[1]] = trueskill.Rating()
                   

for row in results:
    if row[4] == 'left':
        image_scores[row[0]], image_scores[row[1]] = trueskill.rate_1vs1(image_scores[row[0]], image_scores[row[1]])                    
    elif row[4] == 'right':
        image_scores[row[1]], image_scores[row[0]] = trueskill.rate_1vs1(image_scores[row[1]], image_scores[row[0]])
    else:
        image_scores[row[1]], image_scores[row[0]] = trueskill.rate_1vs1(image_scores[row[1]], image_scores[row[0]], drawn=True)

scores = []

for key in list(image_scores):
    scores.append(image_scores[key].mu)
    
mini, maxi = min(scores), max(scores)

coef = 10/(maxi - mini)

images = glob.glob('../voteimages/*.jpg')
plt.ioff()

for key in list(image_scores):
    image_scores[key] = coef * (image_scores[key].mu - maxi) + 10
#    image_scores[key] = image_scores[key].mu
#    plt.figure()
#    img = misc.imread(images[int(key)])
#    plt.imshow(img)
#    plt.title(image_scores[key])
#    plt.savefig('../scores/%s.jpg'%key)

#'''
image_descs = glob.glob('../voteimages/*.txt')

X = []
y = []

for key in list(image_scores):
    y.append(image_scores[key])
    
    image_index = key

    image_index = int(image_index)
    
    path = image_descs[image_index]
    
    desc = np.loadtxt(path)

    X.append(desc)

X = np.array(X)
y = np.array(y)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)

#tuned_parameters = {'kernel': ['rbf','linear'],
#                     'C': [0.01, 0.1, 1, 10, 100, 1000], 'nu': [0.3,0.4,0.5,0.6,0.7,0.8,0.9]}

clf = svm.NuSVR(C=0.01, nu=0.05, kernel='linear')

#grid = GridSearchCV(estimator=clf, param_grid=tuned_parameters)
#
#grid_result = grid.fit(X, y)
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
#    print("%f (%f) with: %r" % (mean, stdev, param))

clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))


#"""
#------------------------------------------------------------------------------
#'''