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

#### GAUSSIAN DISCRIMINANT ANALYSIS ####


def gda(data, target_attr):
    m, n = np.shape(data)
    y = data[:,target_attr]
    x = data[:, [i for i in range(n) if i != target_attr]]
    x_zeros = data[[i for i in range(m) if y[i] == 0], :][:, [j for j in range(n) if j != target_attr]]
    x_ones = data[[i for i in range(m) if y[i] == 1], :][:, [j for j in range(n) if j != target_attr]]

    mu_zeros = np.matrix(np.mean(x_zeros, axis=0))
    mu_ones = np.matrix(np.mean(x_ones, axis=0))
    sigma = np.cov(x.T)

    sigma_norm = np.linalg.norm(sigma)
    sigma_pinv = np.linalg.pinv(sigma)

    model_zeros = lambda x: 1/(((2*np.pi) ** ((n-1)/2.0)) * math.sqrt(sigma_norm)) * math.exp(-1/2 * ((x - mu_zeros) * sigma_pinv * (x - mu_zeros).T)[0,0])
    model_ones = lambda x: 1/(((2*np.pi) ** ((n-1)/2.0)) * math.sqrt(sigma_norm)) * math.exp(-1/2 * ((x - mu_ones) * sigma_pinv * (x - mu_ones).T)[0,0])

    return model_zeros, model_ones


def apply_gda(gda_model, record):
    model_zeros, model_ones = gda_model

    zero_prob = model_zeros(record)
    one_prob = model_ones(record)

    if one_prob >= zero_prob:
        return 1
    else:
        return 0

#### GAUSSIAN DISCRIMINANT ANALYSIS - TESTS ####

accuracies = []

count = 1
for i in range(K):
    testing_data = split_spambase[i]
    training_data = np.array([item for sublist in [x for ind, x in enumerate(split_spambase) if ind != i] for item in sublist])

    gda_model = gda(training_data, 57)

    test_acc = 0.0
    for record in testing_data:
        y_obs = record[57]
        x_obs = np.matrix(record[:57])

        if apply_gda(gda_model, x_obs) == y_obs:
            test_acc += 1

    test_accuracy = test_acc/(np.shape(testing_data)[0])
    print "Spam GDA Accuracy %i: %f" % (count, test_accuracy)
    accuracies.append(test_accuracy)
    count += 1

print "Spam GDA Average Accuracy: %f" % np.mean(accuracies)