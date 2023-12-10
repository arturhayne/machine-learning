## Adaboost
Aims to improve the performance of weak learners (individual models that perform slightly better than random chance) by combining them into a strong learner. AdaBoost focuses more on the examples that are misclassified by the previous weak learners, adapting to the harder-to-learn instances.

### Pros:
 - Can achieve high accuracy with a combination of simple models.
 - Less prone to overfitting compared to individual weak learners.
 - Can be used with a variety of base learners.
### Cons:
 - Sensitive to noisy data and outliers.
 - The training process can be computationally expensive.
 - If the weak learners are too complex or the number of iterations is too high, AdaBoost can be prone to overfitting.

 ## Paramters

 ### Estimator
 The base estimator from which the boosted ensemble is built. Support for sample weighting is required, as well as proper classes_ and n_classes_ attributes. If None, then the base estimator is DecisionTreeClassifier initialized with max_depth=1

 ### Random state
 Controls the random seed given at each estimator at each boosting iteration. Thus, it is only used when estimator exposes a random_state. Pass an int for reproducible output across multiple function calls. 

## Reference
[Adaboost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)