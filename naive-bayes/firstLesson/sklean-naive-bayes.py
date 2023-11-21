import numpy as np

features = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]) # X
labels = np.array([1, 1, 1, 2, 2, 2]) # Y

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
# Giving the training data to lear the patterns
classifier.fit(features, labels)

# what is the label for this particular point (-0.8, -1)? Which class this belongs to?
print(classifier.predict([[-0.8, -1]]))
# Answer: point (-0.8, -1) belogs to class number 1 

classifier_pf = GaussianNB()
classifier_pf.partial_fit(features, labels, np.unique(labels))

print(classifier_pf.predict([[-0.8, -1]]))