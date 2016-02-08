from __future__ import division
__author__ = 'Vishant'
import matplotlib.pyplot as plt
import numpy as np
import math

#### LOADING ALL DATA ####

def add_ones_column(data):
    return [[1] + row for row in data]

with open("housing_train.txt") as housing_train:
    housing_train_data = [line.split() for line in housing_train]

housing_train_data = np.array(housing_train_data).astype(np.float)
housing_train_data_norm = np.copy(housing_train_data)

housing_train_data = np.concatenate((np.ones(shape=(np.shape(housing_train_data)[0], 1)), housing_train_data), axis=1)

with open("housing_test.txt") as housing_test:
    housing_test_data = [line.split() for line in housing_test]

housing_test_data = np.array(housing_test_data).astype(np.float)
housing_test_data_norm = np.copy(housing_test_data)

housing_test_data = np.concatenate((np.ones(shape=(np.shape(housing_test_data)[0], 1)), housing_test_data), axis=1)

with open("perceptronData.txt") as perceptron:
    perceptron_data = [line.split() for line in perceptron]

perceptron_data = np.array(perceptron_data).astype(np.float)
perceptron_data_norm = np.copy(perceptron_data)

perceptron_data = np.concatenate((np.ones(shape=(np.shape(perceptron_data)[0], 1)), perceptron_data), axis=1)

with open("spambase.data") as spambase:
    spambase_data = [line.split(',') for line in spambase]

spambase_data = np.array(spambase_data).astype(np.float)
spambase_data_norm = np.copy(spambase_data)

K = 10
spambase_data = np.concatenate((np.ones(shape=(np.shape(spambase_data)[0], 1)), spambase_data), axis=1)
split_spambase = np.array([spambase_data[i::K] for i in range(K)])

def normalize():

    f = len(housing_train_data_norm[0]) - 1
    join_data = np.concatenate((housing_train_data_norm,housing_test_data_norm))
    mins = np.array([min(join_data[:,i]) for i in range(f)])
    maxs = np.array([max(join_data[:,i]) for i in range(f)])

    housing_train_data_norm[:, :f] = (housing_train_data_norm[:, :f] - mins)/(maxs - mins)
    housing_test_data_norm[:, :f] = (housing_test_data_norm[:, :f] - mins)/(maxs - mins)

    ####=-----------------

    f = len(spambase_data_norm[0]) - 1
    mins = np.array([min(spambase_data_norm[:,i]) for i in range(f)])
    maxs = np.array([max(spambase_data_norm[:,i]) for i in range(f)])

    spambase_data_norm[:, :f] = (spambase_data_norm[:, :f] - mins)/(maxs - mins)

    ####=-----------------

    f = len(perceptron_data_norm[0]) - 1
    mins = np.array([min(perceptron_data_norm[:,i]) for i in range(f)])
    maxs = np.array([max(perceptron_data_norm[:,i]) for i in range(f)])

    perceptron_data_norm[:, :f] = (perceptron_data_norm[:, :f] - mins)/(maxs - mins)

normalize()

housing_train_data_norm = np.concatenate((np.ones(shape=(np.shape(housing_train_data_norm)[0], 1)), housing_train_data_norm), axis=1)
housing_test_data_norm = np.concatenate((np.ones(shape=(np.shape(housing_test_data_norm)[0], 1)), housing_test_data_norm), axis=1)
spambase_data_norm = np.concatenate((np.ones(shape=(np.shape(spambase_data_norm)[0], 1)), spambase_data_norm), axis=1)
split_spambase_norm = np.array([spambase_data_norm[i::K] for i in range(K)])
perceptron_data_norm = np.concatenate((np.ones(shape=(np.shape(perceptron_data_norm)[0], 1)), perceptron_data_norm), axis=1)


def sigmoid(x):
    return 1/(1 + math.exp(-x))


def apply_lin_reg(beta, record):
    np_record = np.array(record)
    return np.dot(np_record, beta)


def apply_log_reg(beta, record):
    np_record = np.array(record)
    return sigmoid(np.dot(np_record, beta))


def mean_squared_error_lr(data, beta, target_attr):
    return np.mean(map(lambda rec: (apply_lin_reg(beta, [val for ind, val in enumerate(rec) if ind != target_attr]) - rec[target_attr])**2, data))


def mean_squared_error_logr(data, beta, target_attr):
        return np.mean(map(lambda rec: (apply_log_reg(beta, [val for ind, val in enumerate(rec) if ind != target_attr]) - rec[target_attr])**2, data))


#### RIDGE LINEAR REGRESSION ####

def linear_regression_ridge(data, target_attr, reg):
    """

    :param data: A dataset
    :param target_attr: The column number representing the target attribute
    :return: A linear regression vector of coefficients
    """
    y = np.matrix([record[target_attr] for record in data])
    x = np.matrix([[val for ind, val in enumerate(record) if ind != target_attr] for record in data])

    b = np.linalg.pinv(x.T * x + reg*np.identity(x.shape[1])) * x.T * y.T

    return b


    #### RIDGE LINEAR REGRESSION - TESTS ####

# housing_lin_reg = linear_regression_ridge(housing_train_data, 14, 1)
# print "Housing Training Error LR: %f" % mean_squared_error_lr(housing_train_data, housing_lin_reg, 14)
# print "Housing Testing Error LR: %f" % mean_squared_error_lr(housing_test_data, housing_lin_reg, 14)
#
# training_errors_lr = []
# testing_errors_lr = []
#
# for i in range(K):
#     testing_data = split_spambase[i]
#     training_data = [item for sublist in [x for ind, x in enumerate(split_spambase) if ind != i] for item in sublist]
#
#     spam_lin_reg = linear_regression_ridge(training_data, 58, 1)
#     training_errors_lr.append(mean_squared_error_lr(training_data, spam_lin_reg, 58))
#     testing_errors_lr.append(mean_squared_error_lr(testing_data, spam_lin_reg, 58))
#
# print "Spam Average Training Error LR: %f" % np.mean(training_errors_lr)
# print "Spam Average Testing Error LR: %f" % np.mean(testing_errors_lr)


#### GRADIENT DESCENT LINEAR REGRESSION ####

def lin_reg_error(x, y, theta):
    summation = 0.0
    for t in range(x.shape[0]):
        summation += ((np.dot(x[t], theta) - y[t])**2)/2

    return summation


def mean_squared_error_lr(data, beta, target_attr):
    return np.mean(map(lambda rec: (apply_lin_reg(beta, [val for ind, val in enumerate(rec) if ind != target_attr]) - rec[target_attr])**2, data))


def linear_regression_grad_desc(data, target_attr, learn_rate, iterations):
    y = np.array([record[target_attr] for record in data])
    x = np.array([[val for ind, val in enumerate(record) if ind != target_attr] for record in data])

    m, n = np.shape(x)
    theta = np.zeros(shape=n)

    for i in range(0, iterations):
        for t in range(m):
            for j in range(len(theta)):
                theta[j] = theta[j] - (learn_rate * (np.dot(theta, x[t]) - y[t]) * x[t,j])

    return theta


    #### GRADIENT DESCENT LINEAR REGRESSION - TESTS ####

# housing_lin_reg_gd = linear_regression_grad_desc(housing_train_data_norm, 14, 0.001, 1000)
# print "Housing Training Error LR GD: %f" % mean_squared_error_lr(housing_train_data_norm, housing_lin_reg_gd, 14)
# print "Housing Testing Error LR GD: %f" % mean_squared_error_lr(housing_test_data_norm, housing_lin_reg_gd, 14)

# training_errors_lr_gd = []
# testing_errors_lr_gd = []
#
# count = 1
# for i in range(K):
#     testing_data = split_spambase_norm[i]
#     training_data = np.array([item for sublist in [x for ind, x in enumerate(split_spambase_norm) if ind != i] for item in sublist])
#
#     spam_lin_reg = linear_regression_grad_desc(training_data, 58, 0.001, 100)
#     training_errors_lr_gd.append(mean_squared_error_lr(training_data, spam_lin_reg, 58))
#     testing_errors_lr_gd.append(mean_squared_error_lr(testing_data, spam_lin_reg, 58))
#
#     print "Spam Training Error LR %i: %f" % (count, mean_squared_error_lr(training_data, spam_lin_reg, 58))
#     print "Spam Testing Error LR %i: %f" % (count, mean_squared_error_lr(testing_data, spam_lin_reg, 58))
#     count += 1
#
# print "Spam Average Training Error LR: %f" % np.mean(training_errors_lr_gd)
# print "Spam Average Testing Error LR: %f" % np.mean(testing_errors_lr_gd)


#### GRADIENT DESCENT LOGISTIC REGRESSION ####

def logistic_regression_grad_desc(data, target_attr, learn_rate, iterations):
    y = np.array([record[target_attr] for record in data])
    x = np.array([[val for ind, val in enumerate(record) if ind != target_attr] for record in data])

    m, n = np.shape(x)
    theta = np.zeros(shape=n)

    for i in range(0, iterations):
        for t in range(m):
            for j in range(len(theta)):
                theta[j] = theta[j] - (learn_rate * (sigmoid(np.dot(theta, x[t])) - y[t]) * x[t,j])

    return theta


    #### GRADIENT DESCENT LOGISTIC REGRESSION - TESTS ####

# training_errors_logr_gd = []
# testing_errors_logr_gd = []
#
# count = 1
# for i in range(K):
#     testing_data = split_spambase_norm[i]
#     training_data = np.array([item for sublist in [x for ind, x in enumerate(split_spambase_norm) if ind != i] for item in sublist])
#
#     spam_log_reg = logistic_regression_grad_desc(training_data, 58, 0.001, 50)
#     training_errors_logr_gd.append(mean_squared_error_logr(training_data, spam_log_reg, 58))
#     testing_errors_logr_gd.append(mean_squared_error_logr(testing_data, spam_log_reg, 58))
#
#     print "Spam Training Error LogR %i: %f" % (count, mean_squared_error_logr(training_data, spam_log_reg, 58))
#     print "Spam Testing Error LogR %i: %f" % (count, mean_squared_error_logr(testing_data, spam_log_reg, 58))
#
#     plt.figure()
#     xs = []
#     ys = []
#
#     for t in [i/100. for i in range(1, 100)]:
#         truePos = falsePos = trueNeg = falseNeg = 0
#
#         for record in testing_data:
#             y = record[58]
#             p = apply_log_reg(spam_log_reg, record[:58])
#
#             if y == 1 and p >= t:
#                 truePos += 1
#             elif y == 0 and p >= t:
#                 falsePos += 1
#             elif y == 0 and p < t:
#                 trueNeg += 1
#             elif y == 1 and p < t:
#                 falseNeg += 1
#
#         truePosRate = truePos/(truePos + falseNeg)
#         falsePosRate = falsePos/(falsePos + trueNeg)
#
#         xs.append(falsePosRate)
#         ys.append(truePosRate)
#
#     print xs
#     print ys
#     auc = np.trapz(ys[::-1], x=xs[::-1])
#     print auc
#     plt.plot(xs, ys)
#     plt.show()
#
#     count += 1
#
# print "Spam Average Training Error LR: %f" % np.mean(training_errors_logr_gd)
# print "Spam Average Testing Error LR: %f" % np.mean(testing_errors_logr_gd)

# testing_data = split_spambase_norm[0]
# training_data = np.array([item for sublist in [x for ind, x in enumerate(split_spambase_norm) if ind != 0] for item in sublist])
# spam_log_reg = logistic_regression_grad_desc(training_data, 58, 0.001, 1000)
# truePos = 0
# falsePos = 0
# trueNeg = 0
# falseNeg = 0
#
# for record in testing_data:
#     y = record[58]
#     p = apply_log_reg(spam_log_reg, record[:58])
#
#     if y == 1 and p >= 0.5:
#         truePos += 1
#     elif y == 0 and p >= 0.5:
#         falsePos += 1
#     elif y == 0 and p < 0.5:
#         trueNeg += 1
#     elif y == 1 and p < 0.5:
#         falseNeg += 1
#
# print "True Positives: %i" % truePos
# print "False Positives: %i" % falsePos
# print "True Negatives: %i" % trueNeg
# print "False Negatives: %i" % falseNeg


# plt.figure()
#
# testing_data = split_spambase_norm[0]
# training_data = np.array([item for sublist in [x for ind, x in enumerate(split_spambase_norm) if ind != 0] for item in sublist])
# x_linreg = []
# y_linreg = []
# for i in range(1, 200):
#     spam_log_reg = logistic_regression_grad_desc(training_data, 58, 0.001, i)
#     truePos = 0
#     falsePos = 0
#     trueNeg = 0
#     falseNeg = 0
#
#     for record in testing_data:
#         y = record[58]
#         p = apply_log_reg(spam_log_reg, record[:58])
#
#         if y == 1 and p >= 0.5:
#             truePos += 1
#         elif y == 0 and p >= 0.5:
#             falsePos += 1
#         elif y == 0 and p < 0.5:
#             trueNeg += 1
#         elif y == 1 and p < 0.5:
#             falseNeg += 1
#
#     x_linreg.append(falsePos/(falsePos + trueNeg))
#     y_linreg.append(truePos/(truePos + falseNeg))
#
# lin_reg_truePos = 141
# lin_reg_falsePos = 12
# lin_reg_trueNeg = 267
# lin_reg_falseNeg = 41
#
# x_linreg = lin_reg_falsePos/(lin_reg_falsePos + lin_reg_trueNeg)
# y_linreg = lin_reg_truePos/(lin_reg_truePos + lin_reg_falseNeg)
# plt.plot(x_linreg, y_linreg)
# # plt.show()
#
# # auc_linreg = np.trapz(y_linreg, x_linreg)
# #
# log_reg_truePos = 154
# log_reg_falsePos = 15
# log_reg_trueNeg = 264
# log_reg_falseNeg = 28
#
# x_logreg = log_reg_falsePos/(log_reg_falsePos + log_reg_trueNeg)
# y_logreg = log_reg_truePos/(log_reg_truePos + log_reg_falseNeg)
# plt.plot(x_logreg, y_logreg)
# plt.show()
#
# auc_logreg = np.trapz(y_logreg,x_logreg)
#
# print "Linear Regression AUC: %.5f" % auc_linreg
# print "Logistic Regression AUC: %.5f" % auc_logreg
#
# plt.show()


#### PERCEPTRON ####

def perceptron(data, target_attr, learn_rate):
    y = np.ones(shape=data.shape[0])
    x = np.array([[val*record[target_attr] for ind, val in enumerate(record) if ind != target_attr] for record in data])

    w = np.random.rand(x.shape[1])

    M = np.array([row for row in x if np.dot(row, w) < 0])
    count = 1
    print "Iteration %i - Total mistakes: %i" % (count, len(M))

    while len(M) > 0:
        for misclass in M:
            w = w + learn_rate*np.array(misclass)

        M = np.array([row for row in x if np.dot(row, w) < 0])
        count += 1
        print "Iteration %i - Total mistakes: %i" % (count, len(M))

    print "Classifier Weights: ", w.T
    return w


def apply_perceptron(percept, record):
    return np.dot(record,percept)


def mean_squared_error_perceptron(data, percept, target_attr):
    return np.mean(map(lambda rec: (apply_perceptron(percept, [val for ind, val in enumerate(rec) if ind != target_attr]) - rec[target_attr])**2, data))


    #### PERCEPTRON - TESTS ####

percept = perceptron(perceptron_data_norm, 5, 0.001)
print "Normalized with threshold: ", -percept[1:]/percept[0]

right = 0
for row in perceptron_data_norm:
    if np.dot(percept, perceptron_data_norm[i,:5])*perceptron_data_norm[i,5] > 0:
        right += 1

print "Perceptron Accuracy: %f" % (right/perceptron_data_norm.shape[0])


#### NEURAL NETWORK AUTOENCODER ####

autoencoder_data = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]])


def sigmoid_single_layer_neural_network(hidden_size, indata, outdata, learn_rate, iterations):
    input_size = indata.shape[0]
    output_size = outdata.shape[0]
    w_in_to_hidden = np.random.rand(hidden_size, input_size)
    w_hidden_to_out = np.random.rand(output_size, hidden_size)

    bias_hidden = np.random.rand(hidden_size)
    bias_output = np.random.rand(output_size)

    for i in range(iterations):
        for ind, record in enumerate(indata):
            i_hidden = np.dot(w_in_to_hidden, record) + bias_hidden
            o_hidden = np.array(map(sigmoid, i_hidden))

            i_output = np.dot(w_hidden_to_out, o_hidden) + bias_output
            o_output = np.array(map(sigmoid, i_output))

            errs_output = np.array([o_output[j]*(1 - o_output[j])*(outdata[ind,j] - o_output[j]) for j in range(output_size)])
            errs_hidden = np.array([o_hidden[j]*(1 - o_hidden[j])*np.dot(errs_output,w_hidden_to_out[:,j]) for j in range(hidden_size)])

            for i in range(hidden_size):
                for j in range(input_size):
                    w_in_to_hidden[i,j] += learn_rate * errs_hidden[i] * record[j]

            for i in range(output_size):
                for j in range(hidden_size):
                    w_hidden_to_out[i,j] += learn_rate * errs_output[i] * o_hidden[j]

            bias_hidden += learn_rate * errs_hidden
            bias_output += learn_rate * errs_output

    return (w_in_to_hidden, w_hidden_to_out, bias_hidden, bias_output)

def apply_neural_network(nn, data):
    w_in_to_hidden, w_hidden_to_out, bias_hidden, bias_output = nn

    i_hidden = np.dot(w_in_to_hidden, data) + bias_hidden
    o_hidden = np.array(map(sigmoid, i_hidden))

    i_output = np.dot(w_hidden_to_out, o_hidden) + bias_output
    o_output = np.array(map(sigmoid, i_output))

    return np.array(o_output)


    #### NEURAL NETWORK AUTOENCODER - TESTS ####

# autoencoder_nn = sigmoid_single_layer_neural_network(3, autoencoder_data, autoencoder_data, 0.1, 20000)
# np.set_printoptions(formatter={'all':lambda x: '{0:.2f}'.format(x)})
#
# for i in range(8):
#     print apply_neural_network(autoencoder_nn, autoencoder_data[i])
#
# np.set_printoptions()