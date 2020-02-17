#Hierarchical Clustering
"""
Created on Mon Oct 14 20:15:11 2019

@author: SHIVAM
"""

#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv('mall.csv')
X = dataset.iloc[:, [3, 4]].values

#Finding the optimal number of clusters
#ward() method tries to minimize the variance within each cluster
import scipy.cluster.hierarchy as sch 
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Number of customers')
plt.ylabel('Euclidean Distance')
plt.show()

#Watch the longest vertical line that does not cut any horizontal line
#Count the number of clusters below that vertical line

#Fitting the model to data
from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y = model.fit_predict(X)

#Visualising the results 
plt.scatter(X[y == 0, 0], X[y == 0, 1], s = 100, c = 'red', label = 'Careful')
plt.scatter(X[y == 1, 0], X[y == 1, 1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(X[y == 2, 0], X[y == 2, 1], s = 100, c = 'cyan', label = 'Target')
plt.scatter(X[y == 3, 0], X[y == 3, 1], s = 100, c = 'green', label = 'Careless')
plt.scatter(X[y == 4, 0], X[y == 4, 1], s = 100, c = 'magenta', label = 'Sensible')
plt.title('Clustered Data')
plt.xlabel('Annual Income')
plt.ylabel('Spendings')
plt.legend()
plt.show()