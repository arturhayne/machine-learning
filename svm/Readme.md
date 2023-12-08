## Support Vector Machines
They work really well in complicated domains where there is a clear margin of separation, but don't peform very well in very large data sets, because the train time happens to be cubic in the size of the data set.
Don't work well with lot noise.

## Parameters
[Parameters](https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html) like gamma, C and kernel, prevents the algorithm produces overfitting.

## How to execute
```
# python svm/author-id/svm_author_id.py
# python svm/terrainData/main.py 
```

## Comparing with naive bayes

Naive Bayes is great for text--it’s faster and generally gives better performance than an SVM for this particular problem.

## References
[SVM](https://scikit-learn.org/stable/modules/svm.html)

## Note
In the exercise with the rbf kernel we had to use `gamma=auto` as mentioned in [link](https://github.com/udacity/ud120-projects/issues/283)

