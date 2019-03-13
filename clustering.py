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

def clusterSize(clusterFunction, nbClusters):
	for num_cluster in range(0,nbClusters):
		data_cluster = X_all_norm[clusterFunction.labels_ == num_cluster]
		print(data_cluster.shape[0])

if __name__ == "__main__":
	nbCl=int(sys.argv[1])
	# read input text and put data inside a data frame
	data = pd.read_csv('base_prospect.csv',sep=',')

	prep  = Preprocessing(data)
	X_all_norm = prep.preprocess_attributs()

	print("------- K-Means ---------")
	k_means=KMeans(n_clusters=nbCl)
	k_means.fit(np.array(X_all_norm))
	clusterSize(k_means,nbCl)

	# clustering = DBSCAN(eps=3, min_samples=2, n_jobs=-1).fit(X_all_norm)
	# print(clustering.labels_)

	print("------- AgglomerativeClustering ---------")
	clustering = AgglomerativeClustering(n_clusters=nbCl, linkage='complete').fit(X_all_norm)
	clusterSize(clustering,nbCl)

	print("------- Birch ---------")
	brc = Birch(branching_factor=50, n_clusters=nbCl, threshold=0.5, compute_labels=True)
	brc.fit(X_all_norm)
	clusterSize(brc,nbCl)