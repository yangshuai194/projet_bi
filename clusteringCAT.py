#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd;
from preprocessing import Preprocessing

# read input text and put data inside a data frame
data = pd.read_csv('base_prospect.csv',sep=',')