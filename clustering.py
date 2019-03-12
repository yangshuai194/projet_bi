#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd;
from preprocessing import Preprocessing
from sklearn. cluster import KMeans
import numpy as np

# read input text and put data inside a data frame
data = pd.read_csv('base_prospect.csv',sep=',')

prep  = Preprocessing(data)
X_all_norm = prep.preprocess_attributs()

k_means=KMeans(n_clusters=4)
k_means.fit(np.array(X_all_norm))

for num_cluster in range(0,4):
    data_cluster = X_all_norm[k_means.labels_ == num_cluster]
    # print(data_cluster)
    print(data_cluster.shape[0])