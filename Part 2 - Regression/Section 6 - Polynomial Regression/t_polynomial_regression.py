# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 15:20:51 2019

@author: st
"""

#polynomial regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset 
dataset = pd.read_csv('Position_Salaries.csv')
'''
#to find out the relationship between the salary and level , we use pyplot--> non -linear relationship
plt.figure()
plt.title("Salary vs. Level")
plt.scatter(dataset.iloc[:, 1],dataset.iloc[:, 2], label='high income low saving',color='r')
plt.show()
'''
#bluffing detector using polynomial regression. for new employee

#X=dataset.iloc[:,1].values--> if we execute it, then we get the vecot array and not the matrix
#X is matrix and y is vector
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values
'''
#splitting of datset --> we have only small amount of data, so not gonna split and put all in training
from sklearn.cross_validation import train_test_split
X_train, X_test,y_train,y_test= train_test_split(X,y, test_size =0.2, random_state=0)
'''
#no need feature scaling
#for comparing linear regression and polynomial
#fitting Linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)
#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
#adds additional polynomial terms to the existing data
#hyperparameter degree changed from 2 to 4 to get perfect graph
poly_reg=PolynomialFeatures(degree=4)
#create X_poly matrix having additional variable.
X_poly= poly_reg.fit_transform(X)
#linear object which include poly_reg object
lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)
#visualise linear regrerssion
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title("Truth v/s Bluff (Linear)")
plt.xlabel("Position Levels")
plt.ylabel("Salary")
plt.show()
#visualising the polynomial regression results
X_grid= np.arange(min(X),max(X),0.1)
X_grid= X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title("Truth v/s Bluff (Polynomial regression)")
plt.xlabel("Position Levels")
plt.ylabel("Salary")
plt.show()
#Predict truth or bluff with the linear regression model
#based on the input, the predictions are created
y_pred=lin_reg.predict([[6.5]])

# Predict truth or bluff with the polymonial regression model
y_pred_poly=lin_reg2.predict(poly_reg.fit_transform([[6.5]])) 
