

###Go to link for data ---> https://www.kaggle.com/c/ieee-fraud-detection/data


import pandas as pd
import numpy as np
from sklearn import linear_model
import seaborn as sns
import matplotlib.pyplot as plt

print('Starting...')
#Import files

t_t = pd.read_csv('train_transaction.csv')
t_i = pd.read_csv('train_identity.csv')
ts_t = pd.read_csv('test_transaction.csv')
ts_i = pd.read_csv('test_identity.csv')

print('Files Loaded')

print('Training Model')

df_train = t_t.merge(t_i, how='outer', on='TransactionID')
df_test = ts_t.merge(ts_i, how='outer', on='TransactionID')

#Visualize 

fig, ax = plt.subplots()
ax = t_t['isFraud'].value_counts(normalize=True).map(lambda x: x * 100).plot.bar()
ax.title.set_text('How many transactions are fraudulent?')

df_train.fillna(value=-999, inplace=True)

features = df_train.drop(labels='isFraud', axis=1)
features = features._get_numeric_data()
target = df_train['isFraud']

test_features = df_test.iloc[:, :]
test_features = test_features._get_numeric_data()
test_features.fillna(value=-999, inplace=True)

#Train model

log_model = linear_model.LogisticRegression(solver='lbfgs')
log_model.fit(features, target)

#Make predictions 

fraud_prediction = log_model.predict(test_features)

print(fraud_prediction)

print('Done')
