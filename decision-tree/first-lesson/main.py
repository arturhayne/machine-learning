#!/usr/bin/python

""" lecture and example code for decision tree unit """

import sys
from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from classifyDT import classify
import os

current_directory = 'decision-tree/first-lesson/'
file_name = "test.png"

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the classify() function in classifyDT is where the magic
### happens--fill in this function in the file 'classifyDT.py'!
clf = classify(features_train, labels_train)

pred = clf.predict(features_test);
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

print("Accuracy:", acc);

prettyPicture(clf, features_test, labels_test)
output_image(os.path.join(current_directory, file_name), "png", open("test.png", "rb").read())
