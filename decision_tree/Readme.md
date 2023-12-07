# Decision Tree Classifier

## Overview

This project implements a decision tree classifier, a popular machine learning algorithm for both classification and regression tasks. The decision tree works by recursively partitioning the dataset based on input features, assigning labels or predicting values at the leaves of the tree.

## Entropy and Information Gain

### Entropy
Entropy measures impurity in a set of examples. It guides the decision tree in determining how to split the data. Entropy ranges from 0 to 1, with 0 indicating perfect homogeneity and 1 indicating maximum impurity.The formula for entropy is:

\[ H(S) = - p_1 \cdot \log_2(p_1) - p_2 \cdot \log_2(p_2) - \ldots - p_k \cdot \log_2(p_k) \]

where \( p_1, p_2, \ldots, p_k \) are the proportions of different classes in the set \( S \). Entropy ranges from 0 to 1, with 0 indicating perfect homogeneity (all examples belong to the same class) and 1 indicating maximum impurity.

### Information Gain
Information gain quantifies the effectiveness of a feature in reducing entropy. It evaluates the reduction in entropy achieved by splitting the data based on a particular feature.

## Strengths

- **Ease of Use:** Decision trees are easy to understand and interpret.
- **Graphical Representation:** They provide a visual representation of decision-making.
- **Interpretability:** Decision trees allow transparent reasoning behind predictions.
- **Handle Non-linear Relationships:** They can model complex, non-linear relationships in the data.

## Weaknesses

- **Prone to Overfitting:** Decision trees can capture noise in training data.
- **Sensitive to Data Variations:** Small changes in data can lead to different tree structures.
- **Global Optima:** They might not find the globally optimal tree structure.
- **Struggles with XOR-like Problems:** Decision trees may struggle with non-axis-aligned decision boundaries.

## Usage Tips

- **Prune the Tree:** Limit the tree's depth to prevent overfitting.
- **Tune Parameters:** Carefully tune hyperparameters, such as the maximum depth.
- **Consider Ensemble Methods:** Use ensemble methods like Random Forests for improved generalization.

## Decision Tree Parameters

### `min_samples_split`

The `min_samples_split` parameter in scikit-learn's decision tree classifier is a hyperparameter that controls the minimum number of samples required to split an internal node. If a node has fewer samples than the specified value, the split is not performed, and the node becomes a leaf. This parameter helps to prevent the tree from becoming too deep and overfitting the training data.

We can adjust the `min_samples_split` parameter to control the granularity of the splits in the decision tree. For example:

```
python
clf = tree.DecisionTreeClassifier(min_samples_split=2)
```

## How to run
```
# python naive-bayes/terrainData/main.py 
```

## Refernce
 - [Decision Tree - scikit-learn](https://scikit-learn.org/stable/modules/tree.html)
