#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from scipy.io import arff
import pandas as pd
from preprocessing import Preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn. cluster import KMeans
import sys
import numpy as np

def clusterSize(clusterFunction, nbClusters):
	for num_cluster in range(0,nbClusters):
		data_cluster = X_all_norm[clusterFunction.labels_ == num_cluster]
		print(data_cluster.shape[0])

def oneShootCLuster(data,clf,nbCl):
	print("Adding cluster column to Data...")
	#Ajout d'une colonne par cluster à data
	for num_cluster in range(0,nbCl):
		name = 'cluster_'+str(num_cluster)
		data[name] = pd.Series(0, index=data.index)
	print("Adding done !")

	cluster_map = pd.DataFrame()
	cluster_map['data_index'] = data.index.values
	cluster_map['cluster'] = k_means.labels_
	print("add cluster value...")
	for index, row in data.iterrows():
		for num_cluster in range(0,nbCl):
			name_Col_Cluster = 'cluster_' + str(num_cluster)
			if index in cluster_map[cluster_map.cluster == num_cluster]:
				data[name_Col_Cluster] = 1
	print("Done !")

def percent(nbLine,size):
	return nbLine*100/size

# Ajoute une colonne correspondant au numéro du cluster
def addClumnCluster(data,clf,nbCl):
	percent = []
	for ipercent in range(0,100):
		percent.append(ipercent)
	old= -1
	data['cluster'] = pd.Series(0, index=data.index)
	cluster_map = pd.DataFrame()
	cluster_map['data_index'] = data.index.values
	cluster_map['cluster'] = k_means.labels_
	print("add cluster value...")
	i = 0
	# pour chaque ligne de data
	for index, row in data.iterrows():
		# pour chaque cluster
		for num_cluster in range(0,nbCl):
			# si l'index de la ligne est aussi présebt dans le cluster
			if index in cluster_map[cluster_map.cluster == num_cluster]:
				data['cluster'] = "cluster " + num_cluster
		# indicateur de progression dans le terminal
		prog = i*100/data.shape[0]
		if prog in percent and prog != old:
			print " " + str(prog) + "%   \r",
			old = prog

		i +=1
	print("Done !")
	return data

if __name__ == "__main__":
	if len(sys.argv) > 1:
		nbCl=int(sys.argv[1])
	else:
		nbCl = 8

	print str(nbCl) + " clusters."
	print("Load data...")
	data = pd.read_csv('base_prospect.csv',sep=',')
	print("Data loaded !")
	print("Preprocessing data...")
	prep  = Preprocessing(data)
	X_all_norm = prep.preprocess_attributs()
	print("Preprocessing done !")

	print("K-Means Starting...")
	k_means=KMeans(n_clusters=nbCl)
	k_means.fit(np.array(X_all_norm))
	print("K-Means done !")

	clusterSize(k_means,nbCl)

	data=addClumnCluster(data,k_means,nbCl)

	print("--------------- data --------------")
	print(data)
