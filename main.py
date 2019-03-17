#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from preprocessing import Preprocessing
from clustering import generate_files
from clustering import type_cluster
from sklearn.exceptions import DataConversionWarning
import warnings
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
pd.options.mode.chained_assignment = None


def accuracy_score(lst_classif,lst_classif_names,X,y):
    for clf,name_clf in zip(lst_classif,lst_classif_names):
        scores = cross_val_score(clf, X, y, cv=5)
        print("Accuracy of "+name_clf+" classifier on cross-validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def confusion_matrix(lst_classif,lst_classif_names,X,y):
    for clf,name_clf in zip(lst_classif,lst_classif_names):
        predicted = cross_val_predict(clf, X, y, cv=5) 
        print("Accuracy of "+name_clf+" classifier on cross-validation: %0.2f" % metrics.accuracy_score(y, predicted))
        print(metrics.confusion_matrix(y, predicted))

if __name__ == "__main__":

	print("----------- Loading data -----------")
	data = pd.read_csv('base_prospect.csv',sep=',')
	print("----------- Loading data DONE -----------")
	
	print("----------- preprocessing -----------")
	prep  = Preprocessing(data)
	X_all_norm = prep.preprocess_attributs()
	print("----------- preprocessing DONE -----------")

	print("----------- clustering -----------")
	X_KMeans=type_cluster(X_all_norm,'KMeans',8,1)
	generate_files(X_KMeans,prep,'KMeans')

	# X_Birch=type_cluster(X_all_norm,'Birch',8,0.3)
	# generate_files(X_Birch,prep,'Birch')
	print("----------- clustering DONE-----------")

	print("----------- calculating performance -----------")
	data = pd.read_csv('base_prospect.csv',sep=',')
	prep  = Preprocessing(data)	
	X_all_norm = prep.preprocess_attributs_balance()
	dummycl = DummyClassifier(strategy="most_frequent")
	gmb = GaussianNB()
	dectree = tree.DecisionTreeClassifier(criterion='entropy', max_features='auto')
	logreg = LogisticRegression(solver="liblinear")

	lst_classif = [dummycl, gmb, dectree, logreg]
	lst_classif_names = ['Dummy', 'Naive Bayes', 'Decision tree', 'Logistic regression']
	accuracy_score(lst_classif,lst_classif_names,X_all_norm,prep.y_rdv)

	print("----------- calculating performance DONE -----------")	

	print("----------- prediction -----------")	
	X_train, X_test, y_train, y_test = train_test_split( X_all_norm, prep.y_rdv, test_size = 0.3, random_state = 100)
	dectree.fit(X_train, y_train)
	y_pred = dectree.predict(X_test)
	print( 'Accuracy of Decision tree classifier on test set : {:.2f}'.format(dectree.score(X_test, y_test)))

	X_train, X_test, y_train, y_test = train_test_split( X_KMeans.drop(columns='cluster'), X_KMeans['cluster'], test_size = 0.3, random_state = 100)
	dectree.fit(X_train, y_train)
	y_pred = dectree.predict(X_test)
	print( 'Accuracy of Cluster on test set : {:.2f}'.format(dectree.score(X_test, y_test)))

	print("----------- prediction DONE -----------")	

