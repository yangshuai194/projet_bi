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
from sklearn.impute import SimpleImputer
from sklearn.exceptions import DataConversionWarning
import warnings
from pandas import ExcelWriter
from openpyxl import load_workbook

warnings.filterwarnings(action='ignore')
pd.options.mode.chained_assignment = None
def clusterSize(clusterFunction, nbClusters):
	for num_cluster in range(0,nbClusters):
		data_cluster = X_all_norm[clusterFunction.labels_ == num_cluster]
		print(data_cluster.shape[0])

def oneShootCLusterOld(data,clf,nbCl):
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
def addClumnClusterOld(data,clf,nbCl):
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

def addClumnCluster(data,clf,nbCl):
	percent = []
	for ipercent in range(0,100):
		percent.append(ipercent)
	old= -1
	cluster_map = pd.DataFrame()
	cluster_map['data_index'] = data.index.values
	cluster_map['cluster'] = k_means.labels_
	print("add cluster value...")
	data['cluster'] = pd.Series(cluster_map['cluster'], index=data.index)
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
	X_all_norm=addClumnCluster(X_all_norm,k_means,nbCl)
	print("--------------- data --------------")
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

		# merged_dept.to_csv('dept_'+str(num_cluster)+'.csv', sep=',',index=False)
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
		# merged_code.to_csv('code_'+str(num_cluster)+'.csv', sep=',',index=False)

		# nb_cluster = lst_cluster[lst_cluster == num_cluster].shape[0]
		# lst_oui.append(round(float(sum_oui.loc[num_cluster,:]/nb_cluster),2))
		# lst_non.append(round(float(sum_non.loc[num_cluster,:]/nb_cluster),2))

	writer.close()
	writer2.close()
