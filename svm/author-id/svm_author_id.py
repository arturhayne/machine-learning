#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("/app/tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

## We have to test with/out C and different kernel
clf = SVC(kernel="rbf", gamma = 'auto', C=10000.)

## training dataset down to 1% of its original size to speed up the process
# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data


clf.fit(features_train, labels_train)
#### store your predictions in a list named pred
pred = clf.predict(features_test);

predict10 = 'Sara' if pred[10] == 0 else 'Cris'
predict26 = 'Sara' if pred[26] == 0 else 'Cris'
predict50 = 'Sara' if pred[50] == 0 else 'Cris'

print('Prediction #10:', predict10);
print('Prediction #26:', predict26);
print('Prediction #50:', predict50);

print('All cris count:', sum(pred == 1))
print('All sara count:', sum(pred == 0))

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

print("Accuracy:", acc);