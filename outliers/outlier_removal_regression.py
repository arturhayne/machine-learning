#!/usr/bin/python3

import random
import numpy
import matplotlib.pyplot as plt
import joblib
import sys
sys.path.append("/app/tools/")
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from outlier_cleaner import outlierCleaner
from class_vis import output_image


### load up some practice data with outliers in it
ages = joblib.load( open("./practice_outliers_ages.pkl", "rb") )
net_worths = joblib.load( open("./practice_outliers_net_worths.pkl", "rb") )



### ages and net_worths need to be reshaped into 2D numpy arrays
### second argument of reshape command is a tuple of integers: (n_rows, n_columns)
### by convention, n_rows is the number of data points
### and n_columns is the number of features
ages       = numpy.reshape( numpy.array(ages), (len(ages), 1))
net_worths = numpy.reshape( numpy.array(net_worths), (len(net_worths), 1))
from sklearn.model_selection import train_test_split
ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages, net_worths, test_size=0.1, random_state=42)

### fill in a regression here!  Name the regression object reg so that
### the plotting code below works, and you can see what your regression looks like


reg = linear_model.Ridge()
reg.fit(ages_train, net_worths_train)

coef = reg.coef_
score = reg.score(ages_test, net_worths_test)
print('Slope: {}'.format(coef))
print('Prediction score on test data: {}'.format(score))


try:
    plt.plot(ages, reg.predict(ages), color="blue")
except NameError:
    pass
plt.scatter(ages, net_worths)
plt.show()


### identify and remove the most outlier-y points
cleaned_data = []
try:
    predictions = reg.predict(ages_train)
    cleaned_data = outlierCleaner( predictions, ages_train, net_worths_train )
except NameError:
    print("Your regression object doesn't exist, or isn't name reg")
    print("Can't make predictions to use in identifying outliers")







### only run this code if cleaned_data is returning data
if len(cleaned_data) > 0:
    ages, net_worths, errors = zip(*cleaned_data)
    ages       = numpy.reshape( numpy.array(ages), (len(ages), 1))
    net_worths = numpy.reshape( numpy.array(net_worths), (len(net_worths), 1))

    ### refit your cleaned data!
    try:
        reg.fit(ages, net_worths)
        plt.plot(ages, reg.predict(ages), color="blue")
        coef = reg.coef_
        score = reg.score(ages_test, net_worths_test)
        print('Slope After cleaning: {}'.format(coef))
        print('Prediction score on test data after cleaning: {}'.format(score))    
    except NameError:
        print("You don't seem to have regression imported/created,")
        print("   or else your regression object isn't named reg")
        print("   either way, only draw the scatter plot of the cleaned data")
    plt.scatter(ages, net_worths)
    plt.xlabel("ages")
    plt.ylabel("net worths")
    plt.show()
    plt.savefig("test.png")
    output_image("test.png", "png", open("test.png", "rb").read())

else:
    print("outlierCleaner() is returning an empty list, no refitting to be done")
