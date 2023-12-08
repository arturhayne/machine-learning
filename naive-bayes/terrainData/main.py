#!/usr/bin/python
import sys
sys.path.append("/app/tools/")
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from sklearn.naive_bayes import GaussianNB

directory = '/app/naive-bayes/terrainData'
file_name = 'terrainDataResult.png'
features_train, labels_train, features_test, labels_test = makeTerrainData()

### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]

classifier = GaussianNB()
clf = classifier.fit(features_train, labels_train)
accuracy = clf.score(features_test, labels_test)
print('Accuracy:', accuracy)

### draw the decision boundary with the text points overlaid
prettyPicture(clf, features_test, labels_test, directory + '/'+ file_name)