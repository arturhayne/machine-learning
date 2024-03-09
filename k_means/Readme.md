## Clustering

Clustering is unsupervised learning.
Algorithm used to classify points by grouping them by proximity on the graph.
We stipulate "match points" (centroids) on the graph to define which group the points belong to. Depending on where these points are, we can obtain different results.

### Important Params
 - n_cluster: Number the clusters in the graph.
 - max_iter: How many max interations to find the cluster.
 - n_init: how many times will be run with different centroids seeds.


## Reference 
[KMeans - sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)