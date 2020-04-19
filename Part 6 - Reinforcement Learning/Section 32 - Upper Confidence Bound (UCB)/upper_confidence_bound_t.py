# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 13:43:09 2020

@author: st
"""
#importing dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
#analysis ads click through rate
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
#we don't have dependent variables
#implementing UCB algorithm
N=10000
d=10
ads_selected=[]
number_of_selections=[0] * d
sums_of_rewards=[0] * d
total_reward=0
for n in range(0,N):
    max_upper_bound=0
    ad=0
    for i in range(0,d):
        if(number_of_selections[i]>0):
            average_reward=sums_of_rewards[i]/number_of_selections[i]
            delta_i=math.sqrt(3/2 * math.log(n+1)/number_of_selections[i])# we are using n+1 bcoz as log(n) function starts from 1 but initial value on n starts from 0
            upper_bound=average_reward+delta_i
        else:
            upper_bound =1e400
        if upper_bound>max_upper_bound :
            max_upper_bound=upper_bound
            ad=i
    ads_selected.append(ad)
    number_of_selections[ad]=number_of_selections[ad]+1
    reward=dataset.values[n,ad]
    sums_of_rewards[ad]=sums_of_rewards[ad]+reward
    total_reward=total_reward+reward
#best add is 5th ad
#visualising results
plt.hist(ads_selected) 
plt.title('histogram of ads selections')
plt.xlabel('ads')
plt.ylabel('num_of_times each add was selected')
plt.show()
