from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

X,Y = make_blobs(n_samples=400, centers=4, 
	cluster_std=0.60, random_state=0)

# print(X)

plt.scatter(X[:,0], X[:,1])
plt.savefig('scatterplot.png')





