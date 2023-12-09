#!/usr/bin/python
import sys
sys.path.append("/app/tools/")
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

directory = 'random_forest/'
file_name = "terrainDataResult.png"

features_train, labels_train, features_test, labels_test = makeTerrainData()

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf = clf.fit(features_train, labels_train);

pred = clf.predict(features_test);
acc = accuracy_score(pred, labels_test)

print("Accuracy:", acc);

### draw the decision boundary with the text points overlaid
prettyPicture(clf, features_test, labels_test, directory + file_name, 'Random Forest', acc)
