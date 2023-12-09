#!/usr/bin/python
import sys
sys.path.append("/app/tools/")
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from sklearn.neighbors import KNeighborsClassifier

directory = '/app/k_nearest_neighbors'
file_name = 'terrainDataResult.png'
features_train, labels_train, features_test, labels_test = makeTerrainData()

classifier = KNeighborsClassifier(n_neighbors=3)
clf = classifier.fit(features_train, labels_train)
accuracy = clf.score(features_test, labels_test)
print('Accuracy:', accuracy)

### draw the decision boundary with the text points overlaid
prettyPicture(clf, features_test, labels_test, directory + '/'+ file_name, 'Nearest neighbors', accuracy)