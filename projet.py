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
from sklearn.exceptions import DataConversionWarning
import warnings
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn. cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFECV 
from sklearn.svm import SVR

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
pd.options.mode.chained_assignment = None

# compute within variance
def within_variance(data,centroids,labels,q):
    res = 0.0
    for k in range(q):
        # get number of instances inside cluster k
        n_k = (labels==k).sum()

        # select rows of data associated with each cluster
        d_k = data[np.where(labels==k)]
        # sum squared distances between each point and its centroid
        sum = 0.0
        for vec_k in d_k:
            sum += np.sum(np.square(vec_k-centroids[k]))

        res += sum
    return res/len(data)


# compute between variance
def between_variance(data,centroids,labels,q):
    center = np.average(data,axis=0)

    res = 0.0
    
    for k in range(q):
        # get number of instances inside cluster k
        n_k = (labels==k).sum()

        # sum squared distances between global centroid and each cluster centroid
        res += n_k * np.sum(np.square(centroids[k]-center))

    return res/len(data)



# compute r square for clustering
def r_square(data,centroids,labels,q):
    v_within = within_variance(data,centroids,labels,q)
    v_between = between_variance(data,centroids,labels,q)
    return v_between/(v_between+v_within)


def purity_score(y_true, y_pred):
    """Purity score

    To compute purity, each cluster is assigned to the class which is most frequent 
    in the cluster [1], and then the accuracy of this assignment is measured by counting 
    the number of correctly assigned documents and dividing by the number of documents.
    We suppose here that the ground truth labels are integers, the same with the predicted clusters i.e
    the clusters index.

    Args:
        y_true(np.ndarray): n*1 matrix Ground truth labels
        y_pred(np.ndarray): n*1 matrix Predicted clusters
    
    Returns:
        float: Purity score
    
    References:
        [1] https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that 
    # we count the actual occurence of classes between two consecutive bin
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner
    
    return accuracy_score(y_true, y_voted_labels)


#correlation_circle
def correlation_circle(df,nb_var,x_axis,y_axis,corvar):
    fig, axes = plt.subplots(figsize=(8,8))
    axes.set_xlim(-1,1)
    axes.set_ylim(-1,1)
    # label with variable names
    for j in range(nb_var):
        # ignore two first columns of df: Nom and Code^Z
        plt.annotate(df.columns[j],(corvar[j,x_axis],corvar[j,y_axis]))
    # axes
    plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
    plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)
    # add a circle
    cercle = plt.Circle((0,0),1,color='blue',fill=False)
    axes.add_artist(cercle)
    plt.savefig('acp_correlation_circle_axes_'+str(x_axis)+'_'+str(y_axis))
    plt.close(fig)

def clean_attributs(data):
    
    rdv_oui = data[data['rdv']==1]
    rdv_non = data[data['rdv']==0]
    random_non_rdv = rdv_non.sample(n=rdv_oui.shape[0], random_state=1)

    data=rdv_oui.append(random_non_rdv,ignore_index=True)

    # list all wrong attributs
    wrong_risque=data.index[data['risque'] == '1-6'].tolist()
    wrong_ca=data.index[data['ca_total_FL'] <= 0].tolist()
    lst_wrongs= wrong_risque+wrong_ca

    # delete row by index
    data = data.drop(data.index[lst_wrongs])
    data["chgt_dir"] = data["chgt_dir"].astype(str)

    # data["rdv"] = data["rdv"].astype(str)
    # data.loc[data['rdv'] == "1" ,'rdv'] = "oui"
    # data.loc[data['rdv'] == "0" ,'rdv'] = "non"

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
	one_hot_vectors=pd.get_dummies(data)
	X_cat_norm=pd.DataFrame(imp_frequent.fit_transform(one_hot_vectors),columns=one_hot_vectors.columns)
	return X_cat_norm


def preprocess_attributs_num(data):
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

# read input text and put data inside a data frame
data = pd.read_csv('base_prospect.csv',sep=',')

# nettoyage, recodage des données anormales
data=clean_attributs(data)

y=data['rdv']
y_code_cr=data['code_cr']
y_dept=data['dept']

data = data.drop(columns="dept")
data = data.drop(columns="code_cr")

# Replace missing values by mean and scale numeric values
data_num = data.select_dtypes(include=['float64','int64']).drop('rdv',axis=1)
# Replace missing values by mean and discretize categorical values
data_cat = data.select_dtypes(exclude=['float64','int64'])

X_cat_norm=preprocess_attributs_cat(data_cat)

X_num_norm=preprocess_attributs_num(data_num)

# X_num_norm.sort_values(by=['effectif'],ascending=False)

X_all_norm=pd.concat([X_cat_norm, X_num_norm],axis=1)

dummycl = DummyClassifier(strategy="most_frequent")
gmb = GaussianNB()
dectree = tree.DecisionTreeClassifier()
logreg = LogisticRegression(solver="liblinear")
svc = svm.SVC(gamma='scale')

lst_classif = [dummycl, gmb, dectree, logreg]
lst_classif_names = ['Dummy', 'Naive Bayes', 'Decision tree', 'Logistic regression']

def accuracy_score(lst_classif,lst_classif_names,X,y):
    for clf,name_clf in zip(lst_classif,lst_classif_names):
        scores = cross_val_score(clf, X, y, cv=5)
        print("Accuracy of "+name_clf+" classifier on cross-validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def confusion_matrix(lst_classif,lst_classif_names,X,y):
    for clf,name_clf in zip(lst_classif,lst_classif_names):
        predicted = cross_val_predict(clf, X, y, cv=5) 
        print("Accuracy of "+name_clf+" classifier on cross-validation: %0.2f" % metrics.accuracy_score(y, predicted))
        print(metrics.confusion_matrix(y, predicted))



# accuracy_score(lst_classif,lst_classif_names,X_cat_norm,y)
# confusion_matrix(lst_classif,lst_classif_names,X_all_norm,y)


# k_means=KMeans(n_clusters=4)
# k_means.fit(np.array(X_all_norm))

# my_dict = {i:np.where(k_means.labels_== i)[0] for i in range(k_means.n_clusters)}

# dictList = []
# for key,value in my_dict.iteritems():
#     tmp=[key,value]
#     dictList.append(tmp)

# for num_cluster in range(0,4):
#     data_cluster = X_all_norm[k_means.labels_ == num_cluster]
#     print(data_cluster)
#     # for item in data_cluster:


# print(k_means.cluster_centers_)
# y_kmeans = k_means.predict(X_num_norm)
# plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
# print(X_num_norm['effectif'])
# plt.scatter(X_num_norm['ca_export_FK'], X_num_norm['endettement'], c=y_kmeans, s=2);
# plt.show()

#hiérarchique ascendante
# fig = plt.figure()
# dendrogram(linkage_matrix,color_threshold=0)
# plt.title ('Hierarchical Clustering Dendrogram (Ward)')
# plt.xlabel('sample index')
# plt.ylabel('distance')
# plt.tight_layout()
# plt.savefig ('k-means_hierarchical_clustering')
# plt.close(fig)
# plt.clf()

# for x in range(0,5):
#     lst_k=range(1,15)
#     Sum_of_squared_distances = []
#     for k in lst_k:
#     	est=KMeans(n_clusters=k)
#     	est.fit(X_all_norm)
#     	Sum_of_squared_distances.append(r_square(np.array(X_all_norm), est.cluster_centers_,est.labels_,k))

#     plt.plot(lst_k, Sum_of_squared_distances, 'bx-')
#     plt.xlabel('k')
#     plt.ylabel('Sum_of_squared_distances')
#     plt.title('Elbow Method For Optimal k')
#     plt.savefig ('k-means_elbow_%s'%x)
#     plt.clf()

# estimator = SVR(kernel="linear")
# selector = RFECV(estimator=estimator, cv=5) 
# selector.fit(X_all_norm, y)
# print("Optimal number of features: %d" % selector.n_features_) 
# print(selector.ranking_)


# acp = PCA(svd_solver='full')
# coord = acp.fit_transform(X_num_norm)
# # plot eigen values
# n = np.size(X_num_norm, 0)
# p = np.size(X_num_norm, 1)
# eigval = float(n-1)/n*acp.explained_variance_
# print(pd.DataFrame(acp.components_))
# fig =plt.figure() 
# plt.plot(np.arange(1,p+1),eigval)
# plt.title("Scree plot")
# plt.ylabel("Eigen values")
# plt.xlabel("Factor number")
# plt.savefig('acp_eigen_values') 
# plt.close(fig)


# sqrt_eigval = np.sqrt(eigval) 
# corvar = np.zeros((p,p))
# for k in range(p):
# 	corvar [:, k] = acp.components_[k,:]*sqrt_eigval[k]

# lines: variables # columns: factors
# plot instances on the first plan( first 2 factors )
# fig , axes = plt.subplots(figsize =(12,12))
# axes.set_xlim(-1,1) 
# axes.set_ylim(-1,1) 
# for i in range(n):
# 	plt.annotate(data_num.index,( coord[i ,0], coord[i ,1]) ) 
# plt.plot([-1,1],[0,0], color='silver',linestyle='-',linewidth=1) 
# plt.plot([0,0],[-1,1], color ='silver' , linestyle ='-', linewidth=1)
# plt.savefig('acp_instances_1st_plan')
# plt.close(fig)

# correlation_circle(data_num,len(data_num.columns),0,1,corvar)
# correlation_circle(data_num,len(data_num.columns),1,2,corvar)
# correlation_circle(data_num,len(data_num.columns),2,3,corvar)

# fig , axes = plt.subplots(figsize =(12,12))
# axes.set_xlim(-6,6) 
# axes.set_ylim(-6,6) 
# for i in range(n):
#     plt.annotate(y.values[i],( coord[i,0], coord[i ,1]) ) 
# plt.title("Acp 1 et 2 facteurs") 
# plt.plot([-6,6],[0,0], color='silver',linestyle='-',linewidth=1) 
# plt.plot([0,0],[-6,6], color ='silver' , linestyle ='-', linewidth=1)
# plt.savefig('acp_instances_1&2_facteurs')
# plt.close(fig)

