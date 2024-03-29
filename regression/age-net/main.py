#!/usr/bin/python
import sys
import matplotlib
matplotlib.use('agg')
sys.path.append("/app/tools/")
import matplotlib.pyplot as plt
from class_vis import prettyPicture, output_image
from sklearn import linear_model

from ages_net_worths import ageNetWorthData

ages_train, ages_test, net_worths_train, net_worths_test = ageNetWorthData()


reg = linear_model.Ridge(alpha=.5)
reg.fit(ages_train, net_worths_train)


plt.clf()
plt.scatter(ages_train, net_worths_train, color="b", label="train data")
plt.scatter(ages_test, net_worths_test, color="r", label="test data")
plt.plot(ages_test, reg.predict(ages_test), color="black")
plt.legend(loc=2)
plt.xlabel("ages")
plt.ylabel("net worths")


plt.savefig("test.png")
output_image("test.png", "png", open("test.png", "rb").read())