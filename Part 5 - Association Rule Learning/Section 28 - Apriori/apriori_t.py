# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 19:32:56 2020

@author: st
"""

#Implementing apriori from scratch-> apriori.py -> define custom made functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from apyori import apriori
#importing dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv',header=None)
transactions=[]
#to prepare list to pass through apriori class -> accepts list of list
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
    
#training Apriori using transactions
    #min_support=3*7/7500-> products purchaced 3 times a day - week -> 7 days
    #lift high > 3
    
rules=apriori(transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)

#visualising dataset
#Apriori model is experimental
results=list(rules)

str(list(results[9][2][0][0])[0])
len(results[9][2][0][0])
output =[]
for i in range(0, len(results)):
    output.append(['Rule:\t' + str(results[i][2][0][0]), 'Effect:\t' + str(results[i][2][0][1]),
                   'Support:\t' + str(results[i][1]), 'Confidence:\t' + str(results[i][2][0][2]),
                   'Lift:\t' + str(results[i][2][0][3])])
    for j in range (0, len(results[i][2][0][0])):
        print(str(i)+' '+str(list(results[i][2][0][0])[j]))
#Change value of food to your choice
food = 'chocolate'
print('Recommended: ')
recommend = []
for i in range(0, len(results)):
    for j in range (0, len(results[i][2][0][0])):
        if food in str(list(results[i][2][0][0])[j]):
            for k in range (0, len(results[i][2][0][1])):
                recommend.append((list(results[i][2][0][1])[k]))
                
recommend = list(dict.fromkeys(recommend)) #removes duplicates
if len(recommend)>0:
    print(recommend)
else:
    print('No recommendations')