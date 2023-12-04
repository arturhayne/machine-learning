import sys
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture, output_image

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl
import os

current_directory = os.getcwd()
file_name = "test.png"

features_train, labels_train, features_test, labels_test = makeTerrainData()


########################## SVM #################################
### we handle the import statement and SVC creation for you here
from sklearn.svm import SVC
# C parameter controls the trade-off between smooth decision boundary and classify training points correclty
clf = SVC(kernel="rbf", C=10.0)


#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data


clf.fit(features_train, labels_train)
#### store your predictions in a list named pred
pred = clf.predict(features_test);

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

print("Accuracy:", acc);
prettyPicture(clf, features_test, labels_test)
output_image(os.path.join(current_directory, file_name), "png", open("test.png", "rb").read())