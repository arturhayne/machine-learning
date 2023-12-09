#!/usr/bin/python
import sys
sys.path.append("/app/tools/")
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from sklearn.ensemble import AdaBoostClassifier

directory = '/app/adaboost'
file_name = 'terrainDataResult.png'
features_train, labels_train, features_test, labels_test = makeTerrainData()

classifier = AdaBoostClassifier(n_estimators=100, random_state=0)
clf = classifier.fit(features_train, labels_train)
accuracy = clf.score(features_test, labels_test)
print('Accuracy:', accuracy)

### draw the decision boundary with the text points overlaid
prettyPicture(clf, features_test, labels_test, directory + '/'+ file_name, 'Adaboost', accuracy)