
from sklearn.exceptions import DataConversionWarning
import warnings
import pandas as pd
from preprocessing import Preprocessing
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn import neighbors
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.tree import export_graphviz
import graphviz

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

def display_tree(X_columns):
	pass

if __name__ == "__main__":
	data = pd.read_csv('base_prospect.csv',sep=',')
	prep  = Preprocessing(data)
	X_all_norm = prep.preprocess_attributs_balance()
	dummycl = DummyClassifier(strategy="most_frequent")
	gmb = GaussianNB()
	dectree = tree.DecisionTreeClassifier()
	logreg = LogisticRegression(solver="liblinear")
	svc = svm.SVC(gamma='scale')
	network=MLPClassifier(solver='lbfgs')
	lst_classif = [dummycl, gmb, dectree, logreg]
	lst_classif_names = ['Dummy', 'Naive Bayes', 'Decision tree', 'Logistic regression']

	clf = dectree.fit(X_all_norm, prep.y_rdv)
	dot_data = tree.export_graphviz(clf, out_file=None)
	graph = graphviz.Source(dot_data) 
	graph.render("rdv")

	# accuracy_score(lst_classif,lst_classif_names,X_all_norm,prep.y_rdv)
	# confusion_matrix(lst_classif,lst_classif_names,X_all_norm,prep.y_rdv)


