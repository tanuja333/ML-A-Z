# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 09:27:13 2020

@author: st
"""

#XGBOOST
#BOOST PERFORMANCE AND HIGH EXECTION SPEED
# you don't need to apply sclaing in Xgboost
# 1. high prformnce
# 2. high execution SPEED
# 3. retains the model
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
#creating dummy variables  for countrry
OneHotEncoder=OneHotEncoder(categorical_features=[1])
x=OneHotEncoder.fit_transform(x).toarray()

#removing dummy variable trap by removing a single variable
x=x[:,1:]

#splitting data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25, random_state=0)

#fitting XGbosst to training set
import xgboost as xgb
from xgboost import XGBClassifier
classifier=xgb.XGBClassifier()
classifier.fit(x_train,y_train)
# #XGBClassifier(base_score=0.5, booster=None, colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
#               importance_type='gain', interaction_constraints=None,
#               learning_rate=0.300000012, max_delta_step=0, max_depth=6,
#               min_child_weight=1, missing=nan, monotone_constraints=None,
#               n_estimators=100, n_jobs=0, num_parallel_tree=1,
#               objective='binary:logistic', random_state=0, reg_alpha=0,
#               reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,
#               validate_parameters=False, verbosity=None)

#making predictions and evaluating model
y_pred=classifier.predict(x_test)
#accuracy
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
acc=accuracy_score(y_test,y_pred)
#K-fold cross validation
#Applying K fold cross validation
from sklearn.model_selection import cross_val_score
#9 fold to train and 1 fold to test
#cv is actually num of fold
#for large dataset j_num -> increases the utilisation of all cpu 
accuracies= cross_val_score(estimator=classifier, X=x_train,y=y_train,cv=10)

#avg accuracies mean
accuracies.mean()#--low bias
#avg std
accuracies.std()#--low variance
#can use grid search on XGBoost

# #implementing grid search after validation
# from sklearn.model_selection import GridSearchCV
# # parameters=[{'C':[1,10,100,1000], 'kernel':['linear']},
# #             {'C':[1,10,100,1000],'kernel':['rbf'],'gamma':[0.5,0.1,0.01,0.001,0.0001]}]
# parameters=[{'C':[1,10,100,1000], 'kernel':['linear']},
#             {'C':[1,10,100,1000],'kernel':['rbf'],'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}]
# #for linear model
# grid_search=GridSearchCV(estimator=classifier,
#                          param_grid=parameters,
#                          scoring='accuracy',n_jobs=-1,cv=10)
# grid_search=grid_search.fit(X_train,y_train)
# best_estimator=grid_search.best_estimator_
# best_accuracy=grid_search.best_score_
# best_params=grid_search.best_params_

# #implementing grid search after validation
from sklearn.model_selection import GridSearchCV

param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='accuracy',n_jobs=4,iid=False, cv=5)
gsearch1.fit(x_train,y_train)
best_estimator=gsearch1.best_estimator_
best_accuracy=gsearch1.best_score_
best_params=gsearch1.best_params_
