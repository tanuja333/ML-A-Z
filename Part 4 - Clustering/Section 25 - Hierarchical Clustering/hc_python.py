# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 11:43:37 2019

@author: st
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset= pd.read_csv("C://Users//st//Desktop//2018_docs//ML//Machine Learning A-Z//Part 4 - Clustering//Section 24 - K-Means Clustering//Mall_Customers.csv")
X= dataset.iloc[:,[3,4]].values
#dendograms for finding out optimal number of clustering
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method = 'ward'))
plt.title('Dendograms')
plt.xlabel('Customers')
plt.ylabel('Euclidian dista`nce')
plt.show()
#fit the heirarchial values to data X
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc=hc.fit_predict(X)
#visualising the clusters
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100, c='red', label='careful')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100, c='green', label='standard')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100, c='blue', label='Target')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100, c='cyan', label='Careless')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100, c='magenta', label='Sensible')
plt.title('Cluster of clients')
plt.xlabel('Annual income k$')
plt.ylabel('Spending score(1-100)')
plt.legend()
plt.show()