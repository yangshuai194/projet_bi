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
from openpyxl import load_workbook
from sklearn.exceptions import DataConversionWarning
import warnings

warnings.filterwarnings(action='ignore')
pd.options.mode.chained_assignment = None
def clusterSize(clusterFunction, nbClusters):
	for num_cluster in range(0,nbClusters):
		data_cluster = X_all_norm[clusterFunction.labels_ == num_cluster]
		print(data_cluster.shape[0])

def addColumnCluster(data,clf,nbCl):
	percent = []
	for ipercent in range(0,100):
		percent.append(ipercent)
	old= -1
	cluster_map = pd.DataFrame()
	cluster_map['data_index'] = data.index.values
	cluster_map['cluster'] = clf.labels_
	print("add cluster value...")
	data['cluster'] = pd.Series(cluster_map['cluster'], index=data.index)
	print("Done !")
	return data


def generate_files(X_all_norm,prep):
	lst_cluster=X_all_norm['cluster']
	lst_rdv=prep.y_rdv.reset_index(drop=True)
	lst_dept=prep.y_dept.reset_index(drop=True)
	lst_code=prep.y_code_cr.reset_index(drop=True)

	cluster_rdv=pd.concat([lst_cluster,lst_rdv],axis=1)
	cluster_dept=pd.concat([lst_cluster,lst_dept],axis=1)
	cluster_code=pd.concat([lst_cluster,lst_code],axis=1)

	cluster_dept_rdv=pd.concat([lst_cluster,lst_dept,lst_rdv],axis=1)
	cluster_code_rdv=pd.concat([lst_cluster,lst_code,lst_rdv],axis=1)

	sum_oui=cluster_rdv.groupby('cluster').apply(lambda x: (x=='oui').sum()).reset_index(drop=True).drop(columns='cluster')
	sum_non=cluster_rdv.groupby('cluster').apply(lambda x: (x=='non').sum()).reset_index(drop=True).drop(columns='cluster')

	writer = pd.ExcelWriter("./generated/dept.xlsx",engine='openpyxl')
	writer2 = pd.ExcelWriter("./generated/code_cr.xlsx",engine='openpyxl')
	writer3 = pd.ExcelWriter("./generated/rdv.xlsx",engine='openpyxl')

	for num_cluster in range(0,8):
		cluster_dept = cluster_dept_rdv[cluster_dept_rdv['cluster']==num_cluster]
		dept_oui = cluster_dept[cluster_dept['rdv']=='oui']

		name_oui = 'cluster_'+str(num_cluster)+'_dept_oui'
		name_non = 'cluster_'+str(num_cluster)+'_dept_non'
		count_dept_1 = dept_oui.groupby(['dept','rdv']).size().to_frame(name_oui).reset_index().drop(columns='rdv')
		count_dept_1[name_oui] = count_dept_1[name_oui].astype('int')
		# count1.to_csv(name_oui+".csv", sep=',',index=False)

		dept_non = cluster_dept_rdv[cluster_dept_rdv['rdv']=='non']
		count_dept_2 = dept_non.groupby(['dept','rdv']).size().to_frame(name_non).reset_index().drop(columns='rdv')
		count_dept_2[name_non] = count_dept_2[name_non].astype('int')
		# count2.to_csv(name_non+'.csv', sep=',',index=False)

		merged_dept = pd.merge(count_dept_1,count_dept_2, on='dept', how='outer')
		merged_dept.loc[merged_dept[name_oui].isna(),name_oui] = 0
		merged_dept.loc[merged_dept[name_non].isna(),name_non] = 0
		merged_dept[name_oui] = merged_dept[name_oui].astype('int')
		merged_dept[name_non] = merged_dept[name_non].astype('int')

		merged_dept.to_excel(writer,sheet_name='dept_'+str(num_cluster))  
		writer.save()

		cluster_code = cluster_code_rdv[cluster_code_rdv['cluster']==num_cluster]
		code_oui = cluster_code[cluster_code['rdv']=='oui']
		name_oui = 'cluster_'+str(num_cluster)+'_code_oui'
		name_non = 'cluster_'+str(num_cluster)+'_code_non'

		count_code_1 = code_oui.groupby(['code_cr','rdv']).size().to_frame(name_oui).reset_index().drop(columns='rdv')
		# count_code_1.to_csv(name_oui+".csv", sep=',',index=False)

		code_non = cluster_code_rdv[cluster_code_rdv['rdv']=='non']
		count_code_2 = code_non.groupby(['code_cr','rdv']).size().to_frame(name_non).reset_index().drop(columns='rdv')
		# count_code_2.to_csv(name_non+".csv", sep=',',index=False)

		merged_code = pd.merge(count_code_1,count_code_2, on='code_cr', how='outer')
		merged_code.loc[merged_code[name_oui].isna(),name_oui] = 0
		merged_code.loc[merged_code[name_non].isna(),name_non] = 0

		merged_code[name_oui] = merged_code[name_oui].astype('int')
		merged_code[name_non] = merged_code[name_non].astype('int')
		merged_code.to_excel(writer2,sheet_name='code_'+str(num_cluster))  
		writer2.save()

		rdv = pd.concat([sum_oui.loc[num_cluster,:],sum_non.loc[num_cluster,:]],axis=1)
		rdv.columns= ['rdv_oui','rdv_non']
		rdv.to_excel(writer3,sheet_name='rdv_'+str(num_cluster))  
		writer3.save()


	writer.close()
	writer2.close()
	writer3.close()

def type_cluster(X_all_norm,clt_name,rate):
	if clt_name=='KMeans':
		print("K-Means Starting...")
		k_means=KMeans(n_clusters=nbCl)
		k_means.fit(np.array(X_all_norm))
		clusterSize(k_means,nbCl)
		print("K-Means done !")
		X_all_norm=addColumnCluster(X_all_norm,k_means,nbCl)
		return X_all_norm

	if clt_name=='AgglomerativeClustering':
		print("------- AgglomerativeClustering ---------")
		X_all_norm = X_all_norm.sample(n=X_all_norm.shape[0]*rate,random_state=1)
		clustering = AgglomerativeClustering(n_clusters=nbCl, linkage='complete').fit(X_all_norm)
		clusterSize(clustering,nbCl)
		print("AgglomerativeClustering done !")
		X_all_norm=addColumnCluster(X_all_norm,clustering,nbCl)
		return X_all_norm

	if clt_name=='Birch':
		print("------- Birch ---------")
		X_all_norm = X_all_norm.sample(n=X_all_norm.shape[0]*rate,random_state=1)
		brc = Birch(branching_factor=50, n_clusters=nbCl, threshold=0.5, compute_labels=True)
		brc.fit(X_all_norm)
		clusterSize(brc,nbCl)
		print("Birch done !")
		X_all_norm=addColumnCluster(X_all_norm,brc,nbCl)
		return X_all_norm

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
	X_all_norm=type_cluster(X_all_norm,'KMeans',0.3)
	generate_files(X_all_norm,prep)

	# nbCl=int(sys.argv[1])
	# # read input text and put data inside a data frame
	# data = pd.read_csv('base_prospect.csv',sep=',')

	# prep  = Preprocessing(data)
	# X_all_norm = prep.preprocess_attributs()

	# print("------- K-Means ---------")
	# k_means=KMeans(n_clusters=nbCl)
	# k_means.fit(np.array(X_all_norm))
	# clusterSize(k_means,nbCl)

	# # clustering = DBSCAN(eps=3, min_samples=2, n_jobs=-1).fit(X_all_norm)
	# # print(clustering.labels_)

	# print("------- AgglomerativeClustering ---------")
	# clustering = AgglomerativeClustering(n_clusters=nbCl, linkage='complete').fit(X_all_norm)
	# clusterSize(clustering,nbCl)

	# print("------- Birch ---------")
	# brc = Birch(branching_factor=50, n_clusters=nbCl, threshold=0.5, compute_labels=True)
	# brc.fit(X_all_norm)
	# clusterSize(brc,nbCl)