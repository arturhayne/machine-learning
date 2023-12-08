#!/usr/bin/python3

import sys
from time import time
from sklearn.naive_bayes import GaussianNB
sys.path.append("/app/tools/")
from email_preprocess import preprocess

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

classifier = GaussianNB()

t0 = time()
classifier.fit(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")

t0 = time()
classifier.predict(features_test)
print("Predicting Time:", round(time()-t0, 3), "s")

accuracy = classifier.score(features_test, labels_test)
print("Accuracy", accuracy)