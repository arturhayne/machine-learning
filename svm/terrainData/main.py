import sys
sys.path.append("/app/tools/")
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

current_directory = os.getcwd()
file_name = "/app/svm/terrainData/terrainDataResult-100000.png"

features_train, labels_train, features_test, labels_test = makeTerrainData()

# C parameter controls the trade-off between smooth decision boundary and classify training points correclty
clf = SVC(kernel="rbf", C=100000.0)

clf.fit(features_train, labels_train)
pred = clf.predict(features_test);

acc = accuracy_score(pred, labels_test)
print("Accuracy:", acc);

prettyPicture(clf, features_test, labels_test, file_name)