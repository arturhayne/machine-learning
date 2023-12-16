## Person of interest

In this section we learned how to explore the dataset and find the features.
We used a dataset with some information from the [Eron Scandal](https://en.wikipedia.org/wiki/Enron_scandal)

The script `person_of_interest\explore_eron_data.py` is to play and show some information from the dataset.

Features x data type:
 - "to/from" fields: text
 - Number of emails sent: numerical
 - Content of emails: text
 - Email timestamp: time series
 - Job title: categorical
 

## How to run
You have to download the [dataset](https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz) and unzip in `app\tools`

```
python person_of_interest/explore_eron_data.py 
```