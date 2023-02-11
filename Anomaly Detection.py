# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 15:48:49 2022

@author: RajaI
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

iris = load_iris()
# for pandas column argument: concat iris['feature_names'] list
# and string list (in this case one string); you can make this anything
# the original dataset would probably call this ['Species']
iris_data = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                         columns=iris['feature_names'] + ['target'])
iris_data


plt.boxplot(iris['data'])
plt.show()

X = iris_data.iloc[:, :-1]
y = iris_data.iloc[:, -1]

pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)
# fit the model for outlier detection
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
# use fit_predict to compute the predicted labels of the training samples

y_pred = clf.fit_predict(X_pca)
n_errors = (y_pred != y).sum()
X_scores = clf.negative_outlier_factor_


lofs_index = np.where(y_pred != 1)
values = X_pca[lofs_index]
plt.title("Local Outlier Factor (LOF)")
plt.scatter(X_pca[:, 0], X_pca[:, 1], color='black', label="Normal")
plt.scatter(values[:, 0], values[:, 1], color='r', label="Outliers")
plt.legend(loc="upper left")
plt.show()


# Setting the data
x = iris_data.iloc[:, 0:3].values

css = []

# Finding inertia on various k values
for i in range(1, 8):
    kmeans = KMeans(n_clusters=i, init='k-means++',
                    max_iter=100, n_init=10, random_state=0).fit(X)
    css.append(kmeans.inertia_)

plt.plot(range(1, 8), css, 'bx-')
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('CSS')
plt.show()

#Applying Kmeans classifier
kmeans = KMeans(n_clusters=3,init = 'k-means++', max_iter = 100, n_init =
                10, random_state = 0)

y_kmeans = kmeans.fit_predict(X_pca2222222222222222222233                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   )

kmeans.cluster_centers_

# Visualising the clusters - On the first two columns
plt.scatter(X_pca[:, 0], X_pca[:, 1], label="Normal")
# Plotting the centroids of the clusters
plt.scatter(values[:, 0], values[:, 1], color='r', label="Outliers")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='black', label='Centroids')
plt.title("K means Clusteering")
plt.legend()

