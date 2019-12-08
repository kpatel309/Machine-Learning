# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd #import and manage datasets

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

# Take care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# Only replace columns that have missing data
imputer.fit(X[:, 1:3]) # Change columns 1 and 2

# Change X
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data since ML works on numbers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
# Categorical data is now numbers, but there is still no relationship
# between the categories. Use dummy variables.
onehotencoder = OneHotEncoder(categorical_features = [0]) # One hot encode column 0 (countries column)
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Splitting the dataset into the Training Set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.2, random_state = 0)

# Feature Scaling. Age feature has widely different range than Salary feature
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
#fit test set to the training set so they have same fit
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) # Do not need to fit test set since sc_X is already fitted to training set
# Don't need to feature scale y since this is a classification problem
