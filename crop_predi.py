#importing the required libraries

from sklearn import metrics, preprocessing, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import warnings

#from __future__ import print_function

warnings.filterwarnings('ignore')

df = pd.read_csv('Karnataka_Crops.csv')
df.head()
df['Production'].fillna(df['Production'].mean(), inplace = True)

encoder = preprocessing.LabelEncoder()

df[['Season','District_Name']] = df[['Season','District_Name']].apply(encoder.fit_transform)

features = df[['District_Name', 'Season','Area','Temperature', 'PH', 'Rainfall', 'Phosphorous','Nitrogen','Potash']]

target = df[['crop']]
labels = df[['crop']]

Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)

acc = []
model = []

RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain,Ytrain)

predicted_values = RF.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('RF')

print("RF's Accuracy is: ", x)
#print(classification_report(Ytest,predicted_values))

# Dump the trained Naive Bayes classifier with Pickle
RF_pkl_filename = 'RandomForest.pkl'

# Open the file to save as pkl file
RF_Model_pkl = open(RF_pkl_filename, 'wb')
pickle.dump(RF, RF_Model_pkl)

# Close the pickle instances
RF_Model_pkl.close()