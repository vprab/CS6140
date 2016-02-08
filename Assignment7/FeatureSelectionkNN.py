__author__ = 'Vishant'
import numpy as np
import random

#### LOADING DATA ####

with open("spambase.data") as spambase:
    spambase_data = [line.split(',') for line in spambase]

spambase_data = np.array(spambase_data).astype(np.float)
for i in range(np.shape(spambase_data)[0]):
    if spambase_data[i][-1] == 0:
        spambase_data[i][-1] = -1

K = 10
split_spambase = np.array([spambase_data[i::K] for i in range(K)])
spambase_data_norm = np.copy(spambase_data)

def normalize():

    f = len(spambase_data_norm[0]) - 1
    mins = np.array([min(spambase_data_norm[:,i]) for i in range(f)])
    maxs = np.array([max(spambase_data_norm[:,i]) for i in range(f)])

    spambase_data_norm[:, :f] = (spambase_data_norm[:, :f] - mins)/(maxs - mins)

normalize()

split_spambase_norm = np.array([spambase_data_norm[i::K] for i in range(K)])


#### RELIEF FEATURE SELECTION ####

def reliefFeatureSelection(train, iterations=100):
    m, n = np.shape(train)
    n_features = n - 1

    zeros_data = [r for r in train if r[-1] == -1]
    ones_data = [r for r in train if r[-1] == 1]

    w = np.zeros(n_features)
    num_iter = 0
    while num_iter < iterations:
        i = random.choice(range(m))
        x_i = train[i][:-1]
        y_i = train[i][-1]

        euclidian_distance = lambda a: np.linalg.norm(a[:-1] - x_i)

        if y_i == -1:
            closest_same = min(zeros_data, key=euclidian_distance)
            closest_diff = min(ones_data, key=euclidian_distance)
        else:
            closest_same = min(ones_data, key=euclidian_distance)
            closest_diff = min(zeros_data, key=euclidian_distance)

        for j in range(n_features):
            w[j] = w[j] - (x_i[j] - closest_same[j])**2 + (x_i[j] - closest_diff[j])**2

    return w

# print reliefFeatureSelection(split_spambase_norm[0])

def kNN(train, rec, k, sim_func=None):
    if sim_func is None:
        sim_func = lambda a, b: -np.linalg.norm(a - b)

    neighbors = sorted(train, key=lambda r: -sim_func(rec, r[:-1]))[0:k]
    return max(set(np.array(neighbors)[:,-1]), key=list(np.array(neighbors)[:,-1]).count)