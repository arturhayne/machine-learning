#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""
import os
import joblib
import sys
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

data_dict = joblib.load(open("../tools/final_project_dataset.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

### your code goes here 
### it's all yours from here forward!  

clf = DecisionTreeClassifier()
clf.fit(features,labels)

print("accuracy all data", clf.score(features,labels))


features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)

print("accuracy trained data", clf.score(features_test, labels_test))

pred = clf.predict(features_test)
print(confusion_matrix(labels_test, pred))

from sklearn.metrics import precision_score, recall_score

y_true = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
y_pred = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

print("Precisão:", precision)
print("Revocação:", recall)

import numpy as np

predictions = np.array([0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1])
true_labels = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0])

true_positives = np.sum((predictions == 1) & (true_labels == 1))

print("True positive numbers:", true_positives)

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

from sklearn.metrics import precision_score

precision = precision_score(true_labels, predictions)

print("Precision:", precision)
