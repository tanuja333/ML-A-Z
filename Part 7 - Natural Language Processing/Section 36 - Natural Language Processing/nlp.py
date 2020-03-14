# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 18:07:47 2019

@author: st
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("C://Users//st//Desktop//2018_docs//ML//Machine Learning A-Z//Part 7 - Natural Language Processing//Section 36 - Natural Language Processing//Restaurant_Reviews.tsv",delimiter='\t',quoting=3)
#cleaning datasets
import re #review
import nltk #natural languagre toolkit
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer#stemming
nltk.download('stopwords')
corpus=[]
for i in range(0, 1000):
  review = re.sub('[^a-zA-Z]', ' ',df['Review'][i])
  review = review.lower()
  review = review.split()
  ps=PorterStemmer()
  review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
  review=' '.join(review)
  corpus.append(review)
#creating bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:, 1].values
# done with converting text into numbers. Applying classification model.
#splitting into training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)
#naive_bayes model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)
y_pred= classifier.predict(X_test)

#cal of metric for selecting best model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score,recall_score,f1_score
cm=confusion_matrix(y_test,y_pred)
acc=accuracy_score(y_test,y_pred)
ps= precision_score(y_test,y_pred)
rs=recall_score(y_test,y_pred)
f1s=f1_score(y_test,y_pred)

#CART model
from sklearn.tree import DecisionTreeClassifier
classifier2= DecisionTreeClassifier(criterion='gini', random_state=0)
classifier2.fit(X_train,y_train)
y_pred2= classifier2.predict(X_test)

#cal of metrix for selecting best model
cm2=confusion_matrix(y_test,y_pred2)
acc2=accuracy_score(y_test,y_pred2)
ps2= precision_score(y_test,y_pred2)
rs2=recall_score(y_test,y_pred2)
f1s2=f1_score(y_test,y_pred2)

#Random_forest_classifier
from sklearn.ensemble import RandomForestClassifier
classifier3=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifier3.fit(X_train,y_train)
y_pred3= classifier3.predict(X_test)

#cal of metrix for selecting best model
cm3=confusion_matrix(y_test,y_pred3)
acc3=accuracy_score(y_test,y_pred3)
ps3= precision_score(y_test,y_pred3)
rs3=recall_score(y_test,y_pred3)
f1s3=f1_score(y_test,y_pred3)