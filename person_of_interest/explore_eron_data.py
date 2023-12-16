#!/usr/bin/python3
import numpy as np
import sys
sys.path.append("/app/tools/")
""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

    PossibleFeatures:
    salary
    to_messages
    deferral_payments
    total_payments
    loan_advances
    bonus
    email_address
    restricted_stock_deferred
    deferred_income
    total_stock_value
    expenses
    from_poi_to_this_person
    exercised_stock_options
    from_messages
    other
    from_this_person_to_poi
    poi
    long_term_incentive
    shared_receipt_with_poi
    restricted_stock
    director_fees
    
    
"""

import joblib

enron_data = joblib.load(open("/app/tools/final_project_dataset.pkl", "rb"))
# keysList = list(enron_data["SKILLING JEFFREY K"])
# [element['id'] for element in enron_data['result']['elements']]
# print(sum(1 for p in enron_data if enron_data[p]["poi"] == 1))
# print(sum(1 for _, data in enron_data.items() if data["poi"] == 1))

# print(enron_data["COLWELL WESLEY"]['from_this_person_to_poi'])

# Jeffrey K Skilling
# print(type(enron_data["SKILLING JEFFREY K"]['salary']))

## print(sum(1 for _, data in enron_data.items() if data["salary"] != 'NaN'))
## print(sum(1 for _, data in enron_data.items() if data["email_address"] != 'NaN'))

nan_people_poi = sum(1 for _, data in enron_data.items() if data["total_payments"] == 'NaN')
total = len(enron_data)

print(100*nan_people_poi/total)