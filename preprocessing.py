#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn import metrics
from scipy.io import arff
import pandas as pd;
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.exceptions import DataConversionWarning
import warnings



warnings.filterwarnings(action='ignore', category=DataConversionWarning)
pd.options.mode.chained_assignment = None


class Preprocessing:
	def __init__(self, data):
		self.data = data

	def clean_attributs(self):
		# list all wrong attributs
		wrong_risque=data.index[data['risque'] == '1-6'].tolist()
		wrong_ca=data.index[data['ca_total_FL'] <= 0].tolist()
		
		lst_wrongs= wrong_risque+wrong_ca

		# delete row by index
		data = data.drop(data.index[lst_wrongs])
		data["chgt_dir"] = data["chgt_dir"].astype(str)
		data["dept"] = data["dept"].astype(str)
		data = data.drop(columns="dept")
		data = data.drop(columns="code_cr")
		return data


	def preprocess_attributs_cat(self):
		# edit null value
		#convert chgt_dir 
		data.loc[data['chgt_dir'] == "1.0" ,'chgt_dir'] = "oui"
		data.loc[data['chgt_dir'] == "0.0" ,'chgt_dir'] = "non"
		data.loc[data['chgt_dir'] == "nan" ,'chgt_dir'] = "vide"

		# type_com
		data.loc[data['type_com'].isna() ,'type_com'] = "com_vide"
		
		imp_frequent=SimpleImputer(strategy='most_frequent')

		# convert categorical values into one-hot vectors
		# and replace NA value to most_frequent
		one_hot_vectors=pd.get_dummies(data)
		X_cat_norm=pd.DataFrame(imp_frequent.fit_transform(one_hot_vectors),columns=one_hot_vectors.columns)
		return X_cat_norm


	def preprocess_attributs_num(self):
		imp_frequent=SimpleImputer(strategy='most_frequent')
		imp_mean=SimpleImputer(strategy='mean')
		var = preprocessing.StandardScaler()

		# ca_export_FK
		xScal = var.fit_transform(data[['ca_export_FK']])
		X_ca_export_norm = pd.DataFrame(xScal,columns=data[['ca_export_FK']].columns)
		X_ca_export_norm = pd.DataFrame(imp_frequent.fit_transform(X_ca_export_norm),columns=data[['ca_export_FK']].columns)

		#evo_risque
		xScal = var.fit_transform(data[['evo_risque']])
		X_evo_risque_norm = pd.DataFrame(xScal)
		X_evo_risque_norm = pd.DataFrame(imp_mean.fit_transform(X_evo_risque_norm),columns=data[['evo_risque']].columns)
		
		data=data.drop(columns=['ca_export_FK', 'evo_risque'])
		xScal = var.fit_transform(data)
		X_num_norm = pd.DataFrame(xScal,columns=data.columns)

		X_num_norm=pd.concat([X_num_norm, X_ca_export_norm,X_evo_risque_norm],axis=1)
		return X_num_norm