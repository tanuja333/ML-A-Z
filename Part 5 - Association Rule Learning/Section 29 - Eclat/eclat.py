# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 20:51:42 2020

@author: st


#implementing eclat using pyfim
import pandas as pd

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
n = len(dataset)
transactions = []
for i in range(0, n):
    transaction = []
    m = len(dataset.values[i])
    for j in range(0, m):
        data = str(dataset.values[i,j])
        if data != "nan":
            transaction.append(data)
            transactions.append(transaction)

results = []
from fim import eclat
rules = eclat(tracts = transactions, zmin = 1)
rules.sort(key = lambda x: x[1], reverse = True)
"""
#using mlxtend
#code from :https://github.com/Nikronic/Machine-Learning-Models/blob/master/Part%205%20-%20Association%20Rule%20Learning/Section%2017%20-%20Eclat/eclat.py
# import libraries
import numpy as np
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv',names=np.arange(1,21))

# Preprocesing data to fit model
transactions = []
for sublist in dataset.values.tolist():
    clean_sublist = [item for item in sublist if item is not np.nan]
    transactions.append(clean_sublist) # remove 'nan' values # https://github.com/rasbt/mlxtend/issues/433

from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_x = pd.DataFrame(te_ary, columns=te.columns_) # encode to onehot

# Train model using Apiori algorithm 
# ref = https://rasbt.github.io/mlxtend/api_subpackages/mlxtend.frequent_patterns/
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
df_sets = apriori(df_x, min_support=0.005, use_colnames=True)
df_rules = association_rules(df_sets,metric='support',min_threshold= 0.005,support_only=True) 
# if you use only "support", it called "ECLAT"
'''
output =[]
for i in range(0, len(df_rules)):
    output.append(['Rule:\t' + str(df_rules[i][2][0][0]), 'Effect:\t' + str(results[i][2][0][1]),
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
'''