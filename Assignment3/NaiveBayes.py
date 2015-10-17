from __future__ import division
__author__ = 'Vishant'
import numpy as np
import math

#### LOADING DATA ####

with open("spambase.data") as spambase:
    spambase_data = [line.split(',') for line in spambase]

spambase_data = np.array(spambase_data).astype(np.float)
spambase_data_norm = np.copy(spambase_data)

K = 10
split_spambase = np.array([spambase_data[i::K] for i in range(K)])

#### NAIVE BAYES ####

