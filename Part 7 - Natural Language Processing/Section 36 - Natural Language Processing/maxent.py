# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 20:02:48 2019

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
def word_features(word):

    return {'items': word}

from nltk import MaxentClassifier
numIterations = 100
algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0]
# needs data set in form of list(tuple(dict, str))
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ',df['Review'][i]) #keeps only letters and spaces
    review = review.lower() #make all letters lowercase
    review = review.split() #split review into individual words
    ps = PorterStemmer() #tool that stems words
    # stems each word in the review and removes those words found in stopwords
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #review = ' '.join(review)
    if(df['Liked'][i]==0):
      result = "negative";
    else:
      result = "positive" 
    for word in review:

        corpus.append((word_features(word), result))

from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(corpus, test_size = 0.20, random_state = 0)
classifier = nltk.MaxentClassifier.train(X_train, algorithm, max_iter=numIterations)
# Predicting the Test set results
y_pred = classifier.classify(word_features("first"))
# Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm=confusion_matrix(y_test,y_pred)
print(nltk.classify.accuracy(classifier, X_test))

TP = cm[1,1]
TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]

Accuracy = (TP + TN) / (TP + TN + FP + FN)

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1_Score = 2 * Precision * Recall / (Precision + Recall)