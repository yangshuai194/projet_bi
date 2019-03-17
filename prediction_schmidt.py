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
import pandas as pd
from preprocessing import Preprocessing
from sklearn.neighbors import KNeighborsClassifier


def accuracy_score(lst_classif,lst_classif_names,X,y):
	for clf,name_clf in zip(lst_classif,lst_classif_names):
	    scores = cross_val_score(clf, X, y, cv=10)
	    print("Accuracy of "+name_clf+" classifier on cross-validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def confusion_matrix(lst_classif,lst_classif_names,X,y):
	for clf,name_clf in zip(lst_classif,lst_classif_names):
	    predicted = cross_val_predict(clf, X, y, cv=10) 
	    print("Accuracy of "+name_clf+" classifier on cross-validation: %0.2f" % metrics.accuracy_score(y, predicted))
	    print(metrics.confusion_matrix(y, predicted))

def logistic(X, y):
	clf = LogisticRegression(solver="liblinear")
	predicted = cross_val_predict(clf, X, y, cv=10)
	print("Accuracy of classifier on cross-validation: %0.2f" % metrics.accuracy_score(y, predicted))
	print(metrics.confusion_matrix(y, predicted))
	print("Prediction :")
	clf.fit(X,y)
	print(clf.predict(X.iloc[1,:]))

if __name__ == "__main__":
	data = pd.read_csv('base_prospect.csv',sep=',')
	prep  = Preprocessing(data)
	X_all_norm = prep.preprocess_attributs_balance()

	dummycl = DummyClassifier(strategy="most_frequent")

	dectree = tree.DecisionTreeClassifier(max_features='auto')
	dectree2 = tree.DecisionTreeClassifier(criterion='entropy', max_features='auto',)
	lst_dectree = [dectree,dectree2]
	lst_dectree_names = ['Decision tree 1','Decision tree 2']

	lst_classif = [dummycl, dectree]
	lst_classif_names = ['Dummy', 'Decision tree']

	confusion_matrix(lst_dectree,lst_dectree_names,X_all_norm,prep.y_rdv)