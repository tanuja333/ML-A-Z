# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 20:28:55 2020

@author: st
"""

#data preprocessing 
#theono library can be ussed on GPU and CPU
#tensorflow - high computation.
#theano and tensorflow combined to get new DL models.
#Keras --> combining of noth tensorflow and theana
#keras similar to scikit in ML
#Part 1 data preprocessing.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset 
dataset = pd.read_csv('Churn_Modelling.csv')

x=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

#categorical variables 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
LabelEncoder_x_1=LabelEncoder()
x[:,1] =LabelEncoder_x_1.fit_transform(x[:,1])
#gender
LabelEncoder_x_2=LabelEncoder()
x[:,2] =LabelEncoder_x_2.fit_transform(x[:,2])
#creating dummy variables  foe countrry
OneHotEncoder=OneHotEncoder(categorical_features=[1])
x=OneHotEncoder.fit_transform(x).toarray()

#removing dummy variable trap by removing a single variable
x=x[:,1:]

#splitting data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25, random_state=0)

#we need to apply feature scaling 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#Building ANN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
  

#making predictions and evaluating model

