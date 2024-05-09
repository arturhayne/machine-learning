#!/usr/bin/python3

import joblib
import numpy
numpy.random.seed(42)
from sklearn import tree
from sklearn.metrics import accuracy_score
from  sklearn.linear_model import Lasso
from sklearn.feature_extraction.text import TfidfVectorizer

### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "/app/tools/word_data_overfit.pkl" 
authors_file = "/app/tools/email_authors_overfit.pkl"

word_data = joblib.load( open(words_file, "rb"))
authors = joblib.load( open(authors_file, "rb") )

### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



### your code goes here

clf = tree.DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

acc = accuracy_score(pred, labels_test)
num_features = len(features_train)

print("Accuracy:", acc)
print("Number of features:", num_features)

feature_importances = clf.feature_importances_
feature_names = labels_train
feature_importance_dict = dict(zip(feature_names, feature_importances))
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

print("Top 10 most powerful features:")
for feature, importance in sorted_features[:10]:
    print(f"Feature: {feature}, Importance: {importance}")


# Getting the feature importances
feature_importances = clf.feature_importances_

# Getting the indices of the top 10 features
top_indices = feature_importances.argsort()[-10:][::-1]

# Getting the corresponding feature names and their scores
feature_names = vectorizer.get_feature_names_out()
top_features_with_scores = [(idx, feature_names[idx], feature_importances[idx]) for idx in top_indices]

print("\nTop 10 most powerful features with their scores:")
for idx, feature, score in top_features_with_scores:
    print(idx, " - ", feature, "-", score)
