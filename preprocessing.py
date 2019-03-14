#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn import metrics
from scipy.io import arff
import pandas as pd;
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import numpy as np
# from sklearn.exceptions import dataConversionWarning
import warnings



# warnings.filterwarnings(action='ignore', category=self.dataConversionWarning)
pd.options.mode.chained_assignment = None


class Preprocessing:

	def __init__(self):
		pass

	def __init__(self, data1):
		self.data = data1

	def getData(self):
		return self.data

	def balance_data(self):
		rdv_oui = self.data[self.data['rdv']==1]
   		rdv_non = self.data[self.data['rdv']==0]
   		self.rdv_non = rdv_non.sample(n=rdv_oui.shape[0], random_state=1)
   		self.data = rdv_oui.append(self.rdv_non,ignore_index=True)

   	def balance_data_rate(self,rate):
		rdv_oui = self.data[self.data['rdv']==1]
   		rdv_non = self.data[self.data['rdv']==0]
   		rate_oui = int(rdv_oui.shape[0] * rate)
		rate_non = int(rdv_non.shape[0] * rate)
   		self.rdv_oui = rdv_non.sample(n=rate_oui, random_state=1)
   		self.rdv_non = rdv_non.sample(n=rate_non, random_state=1)
   		self.data = self.rdv_oui.append(self.rdv_non,ignore_index=True)

   	def balance_data_rate_all(self,rate):
   		rate_int = int(self.data['rdv'].shape[0] * rate)
   		self.data = self.data.sample(n=rate_int, random_state=1).reset_index()

	def clean_attributs(self):
		# list all wrong attributs
		wrong_risque=self.data.index[self.data['risque'] == '1-6'].tolist()
		wrong_ca=self.data.index[self.data['ca_total_FL'] <= 0].tolist()
		lst_wrongs= wrong_risque+wrong_ca

		# delete row by index
		self.data.drop(self.data.index[lst_wrongs], inplace=True)
		self.data["chgt_dir"] = self.data["chgt_dir"].astype(str)

		self.data["rdv"] = self.data["rdv"].astype(str)
		self.data.loc[self.data['rdv'] == "1" ,'rdv'] = "oui"
		self.data.loc[self.data['rdv'] == "0" ,'rdv'] = "non"

		self.data["dept"] = self.data["dept"].astype(str)

		self.y_rdv=self.data['rdv']
		self.y_dept=self.data['dept']
		self.y_code_cr=self.data['code_cr']

		self.data.drop(columns='rdv', inplace=True)
		self.data.drop(columns='dept', inplace=True)
		self.data.drop(columns='code_cr', inplace=True)


	def preprocess_attributs_cat(self):
		# Replace missing values by mean and discretize categorical values
		data_cat = self.data.select_dtypes(exclude=['float64','int64'])
		# edit null value
		#convert chgt_dir 
		data_cat.loc[data_cat['chgt_dir'] == "1.0" ,'chgt_dir'] = "oui"
		data_cat.loc[data_cat['chgt_dir'] == "0.0" ,'chgt_dir'] = "non"
		data_cat.loc[data_cat['chgt_dir'] == "nan" ,'chgt_dir'] = "vide"

		# type_com
		data_cat.loc[self.data['type_com'].isna() ,'type_com'] = "com_vide"
		
		imp_frequent=SimpleImputer(strategy='most_frequent')

		# convert categorical values into one-hot vectors
		# and replace NA value to most_frequent
		one_hot_vectors=pd.get_dummies(data_cat)
		self.X_cat_norm=pd.DataFrame(imp_frequent.fit_transform(one_hot_vectors),columns=one_hot_vectors.columns)
		return self.X_cat_norm

	def preprocess_attributs_num(self):
		# Replace missing values by mean and scale numeric values
		data_num = self.data.select_dtypes(include=['float64','int64'])
		imp_frequent=SimpleImputer(strategy='most_frequent')
		imp_mean=SimpleImputer(strategy='mean')
		var = preprocessing.StandardScaler()

		# ca_export_FK
		xScal = var.fit_transform(data_num[['ca_export_FK']])
		X_ca_export_norm = pd.DataFrame(xScal,columns=data_num[['ca_export_FK']].columns)
		X_ca_export_norm = pd.DataFrame(imp_frequent.fit_transform(X_ca_export_norm),columns=data_num[['ca_export_FK']].columns)

		#evo_risque
		xScal = var.fit_transform(data_num[['evo_risque']])
		X_evo_risque_norm = pd.DataFrame(xScal)
		X_evo_risque_norm = pd.DataFrame(imp_mean.fit_transform(X_evo_risque_norm),columns=data_num[['evo_risque']].columns)
		
		data_num=data_num.drop(columns=['ca_export_FK', 'evo_risque'])
		xScal = var.fit_transform(data_num)
		X_num_norm = pd.DataFrame(xScal,columns=data_num.columns)

		self.X_num_norm=pd.concat([X_num_norm, X_ca_export_norm,X_evo_risque_norm],axis=1)
		return self.X_num_norm

	def preprocess_attributs(self):
		self.clean_attributs()
		self.X_num_norm = self.preprocess_attributs_num()
		self.X_cat_norm = self.preprocess_attributs_cat()
		self.data = pd.concat([X_cat_norm, X_num_norm],axis=1)
		return self.data

	def preprocess_attributs_balance(self):
		self.balance_data()
		self.clean_attributs()
		X_num_norm = self.preprocess_attributs_num()
		X_cat_norm = self.preprocess_attributs_cat()
		self.data = pd.concat([X_cat_norm, X_num_norm],axis=1)
		return self.data

	def preprocess_attributs_clustering(self,rate):
		self.balance_data_rate_all(rate)
		self.clean_attributs()
		X_num_norm = self.preprocess_attributs_num()
		X_cat_norm = self.preprocess_attributs_cat()
		self.data = pd.concat([X_cat_norm, X_num_norm],axis=1)
		return self.data