import pandas as pd
import numpy as np
import seaborn as sns
import os
import glob
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from keras.optimizers import Optimizer
from keras import optimizers

df= pd.read_csv("Karnataka_Crops.csv")

cr = LabelEncoder()
se = LabelEncoder()
dis=LabelEncoder()
df['Season'] = se.fit_transform(df['Season'])
df['crop'] = cr.fit_transform(df['crop'])
df['District_Name'] = dis.fit_transform(df['District_Name'])
df.dropna(inplace=True)

df.drop(['PH','Phosphorous','Nitrogen','Potash'],axis='columns', inplace=True)

y = df['Production']
x = df.drop(['Production'], axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=69)

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
x_trainscaled = scaler.fit_transform(x_train)
x_testscaled = scaler.transform(x_test)

from sklearn.ensemble import RandomForestRegressor

RF=RandomForestRegressor()
RF.fit(x_trainscaled,y_train)

ypredranscaled=RF.predict(x_testscaled)

maerandomscaled = mean_absolute_error(ypredranscaled,y_test)

#print(maerandomscaled)

import pickle
RF_pkl_filename = 'yield.pkl'
# Open the file to save as pkl file
RF_Model_pkl = open(RF_pkl_filename, 'wb')
pickle.dump(RF, RF_Model_pkl)

# RF_pkl_filename = "Pickle_RL_Model.pkl"  

# with open(RF_pkl_filename, 'wb') as file:  
#     pickle.dump(RF, file)
# Close the pickle instances
RF_Model_pkl.close()
