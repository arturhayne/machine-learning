## Supervised Classification

Learning about supervised classification using gaussian naive bayes.
The ideias is make input predictions based in a pre-trained dataset.
This is part of the [Udacity Intro Machine Learning program](https://learn.udacity.com/courses/ud120)

### 
The objective of this exercise is to recreate the decision boundary found in the lesson [video](https://www.youtube.com/watch?v=wpnDwiqTCJA&t=1s), and make a plot that visually shows the decision boundary.

There are three projects:
 - First lesson: just an example how to use sklearn.naive_bayes library to predicit which class a point belongs to.
 - Terrain data: determine if car should go fast or slow based a pre-trained data set of bumpy and grade of the terrain. Executable = studentMain.py. The output is a xy graph (test.png) showing the speed of car.
 - Author id: Idetify the author of some emails after a pre-trained data set. Executable = nb_author_id.py.

All solutions uses:
```
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

# Training
classifier.fit(features_train, labels_train)
# Predicting
classifier.predict(features_test)
accuracy = classifier.score(features_test, labels_test)
```
To classify and predict the inputs.

## References
[Naive GaussianNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
