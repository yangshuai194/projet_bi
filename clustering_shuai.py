#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd;
from preprocessing import Preprocessing
from sklearn. cluster import KMeans
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
import sys
import matplotlib.pyplot as plt
from sklearn.exceptions import DataConversionWarning
import warnings

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
pd.options.mode.chained_assignment = None

# compute within variance
def within_variance(data,centroids,labels,q):
    res = 0.0
    for k in range(q):
        # get number of instances inside cluster k
        n_k = (labels==k).sum()

        # select rows of data associated with each cluster
        d_k = data[np.where(labels==k)]
        # sum squared distances between each point and its centroid
        sum = 0.0
        for vec_k in d_k:
            sum += np.sum(np.square(vec_k-centroids[k]))

        res += sum
    return res/len(data)


# compute between variance
def between_variance(data,centroids,labels,q):
    center = np.average(data,axis=0)

    res = 0.0
    
    for k in range(q):
        # get number of instances inside cluster k
        n_k = (labels==k).sum()

        # sum squared distances between global centroid and each cluster centroid
        res += n_k * np.sum(np.square(centroids[k]-center))

    return res/len(data)



# compute r square for clustering
def r_square(data,centroids,labels,q):
    v_within = within_variance(data,centroids,labels,q)
    v_between = between_variance(data,centroids,labels,q)
    return v_between/(v_between+v_within)


def clusterSize(clusterFunction, nbClusters):
	for num_cluster in range(0,nbClusters):
		data_cluster = X_all_norm[clusterFunction.labels_ == num_cluster]
		print(data_cluster.shape[0])



if __name__ == "__main__":
	# nbCl=int(sys.argv[1])
	# read input text and put data inside a data frame
	data = pd.read_csv('base_prospect.csv',sep=',')

	prep  = Preprocessing(data)
	X_all_norm = prep.preprocess_attributs()

	print("------- K-Means ---------")
	for x in range(0,4):
		lst_k=range(1,15)
		Sum_of_squared_distances = []
		for k in lst_k:
			est=KMeans(n_clusters=k)
			est.fit(X_all_norm)
			Sum_of_squared_distances.append(r_square(np.array(X_all_norm), est.cluster_centers_,est.labels_,k))

		plt.plot(lst_k, Sum_of_squared_distances, 'bx-')
		plt.xlabel('k')
		plt.ylabel('Sum_of_squared_distances')
		plt.title('Elbow Method For Optimal k')
		plt.savefig ('./img/k-means_elbow')
		plt.clf()


	# from sklearn.mixture import GaussianMixture
	# gmm = GaussianMixture(n_components=8).fit_predict(X_all_norm.sample(n=100,random_state=1),prep.y_rdv)

	# k_means=KMeans(n_clusters=nbCl)
	# k_means.fit(np.array(X_all_norm))
	# clusterSize(k_means,nbCl)


	# clustering = DBSCAN(eps=3, min_samples=2, n_jobs=-1).fit(X_all_norm)
	# print(clustering.labels_)

	# print("------- AgglomerativeClustering ---------")
	# clustering = AgglomerativeClustering(n_clusters=nbCl, linkage='complete').fit(X_all_norm)
	# clusterSize(clustering,nbCl)

	# print("------- Birch ---------")
	# brc = Birch(branching_factor=50, n_clusters=nbCl, threshold=0.5, compute_labels=True)
	# brc.fit(X_all_norm)
	# clusterSize(brc,nbCl)