# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 22:33:04 2019

@author: st
"""
#%reset -f
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')
X= dataset.iloc[:,[3,4]].values
#Elbow method
from sklearn.cluster import KMeans
wcss=[]
for i in range (1,11):
  kmeans=KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init= 10, random_state=0)
  kmeans.fit(X)
  #another name for wcss is inertia
  wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('Num of clusters')
plt.ylabel('wcss')
plt.show()
#applying Kmeans algorithm to mall dataset
kmeans=KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init= 10, random_state=0)
ykmeans=kmeans.fit_predict(X)
#visualisation the clusters only 2 D
plt.scatter(X[ykmeans==0,0],X[ykmeans==0,1],s=100, c='red', label='careful')
plt.scatter(X[ykmeans==1,0],X[ykmeans==1,1],s=100, c='green', label='standard')
plt.scatter(X[ykmeans==2,0],X[ykmeans==2,1],s=100, c='blue', label='Target')
plt.scatter(X[ykmeans==3,0],X[ykmeans==3,1],s=100, c='cyan', label='Careless')
plt.scatter(X[ykmeans==4,0],X[ykmeans==4,1],s=100, c='magenta', label='Sensible')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow', label='Centroid')
plt.title('Cluster of clients')
plt.xlabel('Annual income k$')
plt.ylabel('Spending score(1-100)')
plt.legend()
plt.show()