
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

# read input text and put data inside a data frame
data = pd.read_csv('base_prospect.csv',sep=',')

data_null = data[pd.isnull(data).any(axis=1)]
data_null.to_csv("data_null.csv", sep='\t')

# print(data.head())

# print nb of instances and features
# print(data.shape
# print feature types
# print(data.dtypes)

# print balance between classes
# print(data.groupby('fruit_name').size())
