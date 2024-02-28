#!/usr/bin/python3
import os
import joblib
import sys
import matplotlib.pyplot as plt
import pandas as pd
sys.path.append("/app/tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = joblib.load(open("/app/tools/final_project_dataset.pkl", "rb"))
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below
plt.scatter(data[:,0], data[:,1])
plt.xlabel("salary")
plt.ylabel("bonus")

df = pd.DataFrame(data_dict)
df.loc['salary', :] = pd.to_numeric(df.loc['salary', :], errors='coerce')
df.loc['bonus', :] = pd.to_numeric(df.loc['bonus', :], errors='coerce')

print(df.loc['salary', :].idxmax())

# Remove Total
data_dict.pop('TOTAL', 0)

data = featureFormat(data_dict, features)

plt.scatter(data[:,0], data[:,1])
plt.xlabel("salary")
plt.ylabel("bonus")
print([name for name in df.columns if df.loc['salary', name] > 10**6 and df.loc['bonus', name] > 5*10**6])


