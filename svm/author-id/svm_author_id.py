#!/usr/bin/python3

import sys
sys.path.append("/app/tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

features_train, features_test, labels_train, labels_test = preprocess()

## We have to test with/out C and different kernel
clf = SVC(kernel="rbf", gamma = 'auto', C=10000.)

## training dataset down to 1% of its original size to speed up the process
# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

clf.fit(features_train, labels_train)
pred = clf.predict(features_test);

predict10 = 'Sara' if pred[10] == 0 else 'Cris'
predict26 = 'Sara' if pred[26] == 0 else 'Cris'
predict50 = 'Sara' if pred[50] == 0 else 'Cris'

print('Prediction #10:', predict10);
print('Prediction #26:', predict26);
print('Prediction #50:', predict50);

print('All cris count:', sum(pred == 1))
print('All sara count:', sum(pred == 0))

acc = accuracy_score(pred, labels_test)

print("Accuracy:", acc);