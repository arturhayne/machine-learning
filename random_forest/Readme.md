## Random Forest
Random Forest builds multiple decision trees during the training phase and combines their predictions to create a more robust and accurate model.

## Pros
 - Generally provides high accuracy and is robust to overfitting.
 - Effective for high-dimensional data and a large number of features.
 - Handles missing values well.

## Cons
 - The model interpretation may be challenging compared to simpler models like decision trees.
 - The model size can be relatively large, especially with a large number of trees.
 - Training time can be longer than some other models, particularly with a large number of trees and features.

 ## Paramters
 ### max depth
 The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

### random_state
Controls both the randomness of the bootstrapping of the samples used when building trees (if bootstrap=True) and the sampling of the features to consider when looking for the best split at each node (if max_features < n_features)


## Reference
[Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)