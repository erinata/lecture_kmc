from sklearn.datasets.samples_generator import make_moons
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

X,Y = make_moons(n_samples=400, noise =0.05, random_state=0)

# print(X)

plt.scatter(X[:,0], X[:,1])
plt.savefig('scatterplot.png')


kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
kmeans_results = kmeans.predict(X)

print(kmeans_results)

plt.scatter(X[:, 0], X[:, 1], c=kmeans_results)
plt.savefig('scatterplot_color.png')

















