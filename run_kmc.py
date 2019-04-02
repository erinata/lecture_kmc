from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

X,Y = make_blobs(n_samples=400, centers=4, 
	cluster_std=0.80, random_state=0)

# print(X)

plt.scatter(X[:,0], X[:,1])
plt.savefig('scatterplot.png')

def run_kmeans(n):
	kmeans = KMeans(n_clusters=n)
	kmeans.fit(X)
	kmeans_results = kmeans.predict(X)

	print(kmeans_results)

	plt.scatter(X[:, 0], X[:, 1], c=kmeans_results)
	plt.savefig('scatterplot_color.png')

run_kmeans(4)

















