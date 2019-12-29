# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:20:46 2019

@author: st
"""
#Multiple Linear Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset=pd.read_csv('50_Startups.csv')
X= dataset.iloc[:, :-1].values
y= dataset.iloc[:, -1].values
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder changes from string to numbers tehn on numbers only the onehotencoder will be used.
labelencoder_X=LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])
#onehotencoder for encoding numeric values
onehotencoder=OneHotEncoder(categorical_features=[3])
#encoding dependent is not necessary bcoz salary is not categorical 
X=onehotencoder.fit_transform(X).toarray()  
#Avoiding the dummy variable trap by removing 1 category remainder
X=X[:, 1:]
# splitting the dataset into training and dataset 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y, test_size=0.2,random_state=0)
#fitting multiiple linear regression to our training set 
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
#Prediction of test set results
y_pred= regressor.predict(X_test)

#building the optimal model using backward elimination
import statsmodels.formula.api as sm #compute the values and evaluate statistical significance for the independent variables
#to make X0=1 , we are adding 1st column as 1 for all trhe values of X
#because the eqn doesn't recognise as the constant
X= np.append(arr=np.ones((50,1)).astype(int),values = X,axis=1) #axis =1 for adding columns   =)

#backward elimination
'''
X_opt = X[:,[0, 1, 2, 3, 4, 5]]
#ordinay least squares
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#removing index 2 as it has more p value compared to 0.05 significance level
X_opt = X[:,[0, 1, 3, 4, 5]]
#ordinay least squares
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#removing index 1 as it has more p value compared to 0.05 significance level
X_opt = X[:,[0, 3, 4, 5]]
#ordinay least squares
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#removing index 4 as it has more p value compared to 0.05 significance level
X_opt = X[:,[0, 3, 5]]
#ordinay least squares
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#removing index 5 as it has more p value compared to 0.05 significance level
X_opt = X[:,[0, 3]]
#ordinay least squares
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#Splitting the optimised data set X_opt into training and test set
X_train_opt,X_test_opt,y_train,y_test= train_test_split(X_opt,y, test_size=0.2,random_state=0)
#fitting multiple linear regression to our optimised training set 
regressor_opt=LinearRegression()
regressor_opt.fit(X_train_opt,y_train)
#optimised Prediction of test set results
y_pred_opt= regressor_opt.predict(X_test_opt)
'''
#Automated backward Elimination with P values and Adjusted R Squared values
#with P values
'''
import statsmodels.formula.api as sm
SL=0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
def backwardElimination(x,sl):
  numVars=len(x[0])
  for i in range(0,numVars):
    regressor_OLS = sm.OLS(y,x).fit()
    maxVar = max(regressor_OLS.pvalues).astype(float)
    if maxVar > sl:
      for j in range(0,numVars - i):
        if(regressor_OLS.pvalues[j].astype(float) == maxVar):
          x = np.delete(x,j,1)
  regressor_OLS.summary()
  return x


X_modeled = backwardElimination(X_opt,SL)
'''
#P values along with Adjusted R squared
'''

'''
SL=0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
def backwardElimination(x, SL):
  numVars=len(x[0])
  #temp = np.zeros((50,6)).astype(int)
  for i in range(0,numVars):
    regressor_OLS= sm.OLS(y,x).fit()
    maxVar = max(regressor_OLS.pvalues).astype(float)
    adjR_before= regressor_OLS.rsquared_adj.astype(float)
    if maxVar > SL:
      for j in range(0, numVars - i):
        if (regressor_OLS.pvalues[j].astype(float) == maxVar):
          #temp[:,j] = x[:, j]
          x_temp = np.delete(x, j, 1)
          tmp_regressor = sm.OLS(y, x_temp).fit()
          adjR_after = tmp_regressor.rsquared_adj.astype(float)
          
          if (adjR_before >= adjR_after):
            #x_rollback = np.hstack((x, temp[:,[0,j]]))
            #x_rollback = np.delete(x_rollback, j, 1)
            print (regressor_OLS.summary())
            return x
          else:
            x= x_temp
            continue
  regressor_OLS.summary()
  return x

X_Modeled = backwardElimination(X_opt, SL)