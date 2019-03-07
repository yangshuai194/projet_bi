#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from scipy.io import arff
import pandas as pd;
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None

def clean_attributs(data):
	# list all wrong attributs
	wrong_risque=data.index[data['risque'] == '1-6'].tolist()
	wrong_ca=data.index[data['ca_total_FL'] <= 0].tolist()
	
	lst_wrongs= wrong_risque+wrong_ca

	# delete row by index
	data = data.drop(data.index[lst_wrongs])
	data["chgt_dir"] = data["chgt_dir"].astype(str)
	data["dept"] = data["dept"].astype(str)

	return data


def preprocess_attributs_cat(data):
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
	onehotvectors=pd.get_dummies(data)
	X_cat_norm=pd.DataFrame(imp_frequent.fit_transform(dum))

	return X_cat_norm


def preprocess_attributs_num(data):
	imp_mean=SimpleImputer(strategy='mean')
	var = preprocessing.StandardScaler()

	# ca_export_FK
	xScal = var.fit_transform(data[['ca_export_FK']])
	X_ca_export_norm = pd.DataFrame(xScal)
	X_ca_export_norm = pd.DataFrame(imp_mean.fit_transform(X_ca_export_norm))

	#evo_risque
	xScal = var.fit_transform(data[['evo_risque']])
	X_evo_risque_norm = pd.DataFrame(xScal)
	X_evo_risque_norm = pd.DataFrame(imp_mean.fit_transform(X_evo_risque_norm))
	
	data.drop(columns=['ca_export_FK', 'evo_risque'])
	xScal = var.fit_transform(data)
	X_num_norm = pd.DataFrame(xScal)

	X_num_norm=pd.concat([X_num_norm, X_ca_export_norm,X_evo_risque_norm],axis=1)

	return X_num_norm

# read input text and put data inside a data frame
data = pd.read_csv('base_prospect.csv',sep=',')

# print(data.index[data['endettement'] <0].shape[0])
# print(data['endettement'].quantile([0.25,0.1]))

# nettoyage, recodage des données anormales
data=clean_attributs(data)

y=data['rdv']

# Replace missing values by mean and scale numeric values
data_num = data.select_dtypes(include=['float64','int64']).drop('rdv',axis=1)
# Replace missing values by mean and discretize categorical values
data_cat = data.select_dtypes(exclude=['float64','int64'])


X_cat_norm=preprocess_attributs_cat(data_cat)
X_num_norm=preprocess_attributs_num(data_num)


# data_null = data[pd.isnull(data).any(axis=1)]
