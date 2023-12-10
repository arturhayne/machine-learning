## K nearest neighbors
k-NN is a versatile algorithm used in various fields, including pattern recognition, image processing, and recommendation systems. It's particularly effective in situations where the decision boundary is complex and not easily captured by a simple mathematical model. The "k" in k-NN refers to the number of nearest neighbors to consider. The algorithm identifies the k instances with the smallest distances to the new instance.

### Pros:
 - Simple and easy to implement.
 - No training phase, which can be advantageous for dynamic datasets.
### Cons:
 - Computationally expensive during the prediction phase, especially with large datasets.
 - Sensitive to irrelevant or redundant features.
 - The choice of distance metric and k value requires careful consideration.


## Reference
[K nearest neighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)