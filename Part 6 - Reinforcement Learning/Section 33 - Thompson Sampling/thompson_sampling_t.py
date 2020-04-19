# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 14:15:10 2020

@author: st
"""

#Thompson sampling
#UBC rewards-2178
#random sampling rewards-1200
#Thompson sampling rewards - 2624
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
#analysis ads click through rate
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
#we don't have dependent variables
#implementing UCB algorithm
N=10000
d=10
ads_selected=[]
# step1
numbers_of_rewards_1= [0] * d
numbers_of_rewards_0= [0] * d
total_reward=0
for n in range(0,N):
    max_random=0
    ad=0
    for i in range(0,d):
        random_beta=random.betavariate(numbers_of_rewards_1[i]+1,numbers_of_rewards_0[i]+1)
        if random_beta>max_random :
            max_random=random_beta
            ad=i
    ads_selected.append(ad)
    reward=dataset.values[n,ad]
    if reward==1:
        numbers_of_rewards_1[ad]=numbers_of_rewards_1[ad]+1
    else:
        numbers_of_rewards_0[ad]=numbers_of_rewards_0[ad]+1
    total_reward=total_reward+reward
#best add is 5th ad
#visualising results
plt.hist(ads_selected) 
plt.title('histogram of ads selections')
plt.xlabel('ads')
plt.ylabel('num_of_times each add was selected')
plt.show()
