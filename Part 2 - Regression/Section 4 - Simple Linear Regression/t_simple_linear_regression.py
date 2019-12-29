# -*- coding: utf-8 -*-
"""
Created on Sat May 25 13:28:21 2019

@author: st
"""
#Simple linear regression
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing dataset
dataset = pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:, 1].values
#splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=1/3,random_state=0)
#Fitting Simple linear regresion to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
#predicting the test set results
y_pred= regressor.predict(X_test)
#visualising the training set results
plt.scatter(X_train,y_train, color='red')#real values
plt.plot(X_train,regressor.predict(X_train),color='blue')#regression line
plt.title('Salary vs Experience(Training set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()
#visualising the test set results
plt.scatter(X_test,y_pred, color='red')#predicted values
plt.plot(X_train,regressor.predict(X_train),color='blue')#regression line
plt.title('Salary vs Experience(Training set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()