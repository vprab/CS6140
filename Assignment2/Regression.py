from __future__ import division
__author__ = 'Vishant'
import numpy as np
import math

### LOADING ALL DATA

def add_ones_column(data):
    return [[1] + row for row in data]

with open("housing_train.txt") as housing_train:
    housing_train_data = [line.split() for line in housing_train]

# housing_train_data = add_ones_column([[float(x) for x in y] for y in housing_train_data])

with open("housing_test.txt") as housing_test:
    housing_test_data = [line.split() for line in housing_test]

# housing_test_data = add_ones_column([[float(x) for x in y] for y in housing_test_data])

housing_train_data = np.array(housing_train_data).astype(np.float)
housing_test_data = np.array(housing_test_data).astype(np.float)

with open("perceptronData.txt") as perceptron:
    perceptron_data = [line.split() for line in perceptron]

perceptron_data = add_ones_column([[float(x) for x in y] for y in perceptron_data])

with open("spambase.data") as spambase:
    spambase_data = [line.split(',') for line in spambase]

K = 10
spambase_data = add_ones_column([[float(x) for x in y] for y in spambase_data])
split_spambase = [spambase_data[i::K] for i in range(K)]


def normalize():
    f = len(housing_train_data[0]) - 1
    mins = np.array([min(housing_train_data[:,i]) for i in range(f)])
    maxs = np.array([max(housing_train_data[:,i]) for i in range(f)])

    housing_train_data[:, :f] = (housing_train_data[:, :f] - mins)/(maxs - mins)

    return (mins, maxs)

normalize()
housing_train_data = np.concatenate((np.ones(shape=(np.shape(housing_train_data)[0], 1)), housing_train_data), axis=1)

# def normalize(train, test):
#     combined = train + test
#     mins = [min([combined[i][j] for i in range(len(combined))]


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

def gradientDescent(data, target_attr, learn_rate, iterations):
    y = np.array([record[target_attr] for record in data])
    x = np.array([[val for ind, val in enumerate(record) if ind != target_attr] for record in data])

    m, n = np.shape(x)
    theta = np.ones(shape=n)

    for i in range(0, iterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y

        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = np.dot(x.T, loss) / m
        # update
        theta = theta - learn_rate * gradient
    return theta


def grad_desc_step(x, y, theta, learn_rate):
    theta_next = np.copy(theta)

    for j in range(len(theta)):
        summation = 0
        for t in range(x.shape[0]):
            summation += ((x[t]*theta)[0,0] - y[t][0,0])*x[t,j]

        theta_next[j] = (theta[j] - learn_rate*summation)[0]

    # theta_next = theta - learn_rate*x.T*(x * theta - y)
    return theta_next


def lin_reg_error(x, y, theta):
    summation = 0.0
    for t in range(x.shape[0]):
        summation += ((np.dot(x[t], theta) - y[t])**2)/2

    return summation


def grad_desc_step_stochastic(x, y, theta, learn_rate):
    theta_next = np.copy(theta)

    for t in range(x.shape[0]):
        for j in range(len(theta)):
            theta_next[j] = theta_next[j] - learn_rate*(np.dot(x[t], theta_next) - y[t])*x[t,j]

    return theta_next


def linear_regression_grad_desc(data, target_attr, learn_rate):
    y = np.array([record[target_attr] for record in data])
    x = np.array([[val for ind, val in enumerate(record) if ind != target_attr] for record in data])

    m, n = np.shape(x)
    theta = np.ones(shape=n)

    # theta_next = np.copy(theta)

    # theta_next = theta - learn_rate*x.T*(x * theta - y)

    # for j in range(theta):
    #
    #     summation = 0
    #     for t in range(x.shape[0]):
    #         summation += (theta*x[t] - y[t])*x[t][j]
    #
    #     theta_next[j] = theta[j] - learn_rate*summation

    theta_next = grad_desc_step_stochastic(x, y, theta, learn_rate)

    J = lambda w: lin_reg_error(x, y, w)
    count = 1

    # while abs(J(theta_next) - J(theta))[0,0] > 0.1:
    #     theta_new = np.copy(theta_next)
    #     theta_next_new = np.copy(grad_desc_step(x, y, theta_next, learn_rate))
    #
    #     theta = np.copy(theta_new)
    #     theta_next = np.copy(theta_next_new)
    #     count += 1

    while abs(J(theta_next) - J(theta)) > 0.1:
        theta_new = np.copy(theta_next)
        theta_next_new = np.copy(grad_desc_step_stochastic(x, y, theta_next, learn_rate))

        theta = np.copy(theta_new)
        theta_next = np.copy(theta_next_new)

    # for t in range(x.shape[0]):
    #     for j in range(len(theta)):
    #         theta[j] = theta[j] - learn_rate*(x[t]*theta - y[t])*x[t,j]

    # for i in range(100):
    #     for t in range(x.shape[0]):
    #         for j in range(len(theta)):
    #             theta[j] = theta[j] - learn_rate*(x[t]*theta - y[t])*x[t,j]

    return theta


def apply_lin_reg(beta, record):
    """

    :param beta: A linear regression vector of coefficients
    :param record: A record of data
    :return: The result of applying the given vector to the record
    """
    np_record = np.array(record)
    return np.dot(np_record, beta)


def mean_squared_error_lr(data, beta, target_attr):
    """

    :param data: A dataset
    :param beta: A linear regression vector of coefficients
    :param target_attr: The column number representing the target attribute
    :return: The MSE between the results of running the linear regression vector on the dataset and the actual target
    values
    """
    return np.mean(map(lambda rec: (apply_lin_reg(beta, [val for ind, val in enumerate(rec) if ind != target_attr]) - rec[target_attr])**2, data))

# housing_lin_reg = linear_regression_ridge(housing_train_data, 14, 1)
# print "Housing Training Error LR: %f" % mean_squared_error_lr(housing_train_data, housing_lin_reg, 14)
# print "Housing Testing Error LR: %f" % mean_squared_error_lr(housing_test_data, housing_lin_reg, 14)

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

housing_lin_reg_gd = linear_regression_grad_desc(housing_train_data, 14, 1)
print "Housing Training Error LR GD: %f" % mean_squared_error_lr(housing_train_data, housing_lin_reg_gd, 14)
# print "Housing Testing Error LR GD: %f" % mean_squared_error_lr(housing_test_data, housing_lin_reg_gd, 13)

# housing_lin_reg_gd = gradientDescent(housing_train_data, 14, 0.0001, 10000)
# print "Housing Training Error LR GD: %f" % mean_squared_error_lr(housing_train_data, housing_lin_reg_gd, 14)
# print "Housing Testing Error LR GD: %f" % mean_squared_error_lr(housing_test_data, housing_lin_reg_gd, 14)

print map(lambda rec: (apply_lin_reg(housing_lin_reg_gd, [val for ind, val in enumerate(rec) if ind != 14])), housing_train_data)

# training_errors_lr = []
# testing_errors_lr = []
#
# for i in range(K):
#     testing_data = split_spambase[i]
#     training_data = [item for sublist in [x for ind, x in enumerate(split_spambase) if ind != i] for item in sublist]
#
#     spam_lin_reg = linear_regression_grad_desc(training_data, 58, 1)
#     training_errors_lr.append(mean_squared_error_lr(training_data, spam_lin_reg, 58))
#     testing_errors_lr.append(mean_squared_error_lr(testing_data, spam_lin_reg, 58))
#
# print "Spam Average Training Error LR GD: %f" % np.mean(training_errors_lr)
# print "Spam Average Testing Error LR GD: %f" % np.mean(testing_errors_lr)

def perceptron(data, target_attr, learn_rate):
    y = np.matrix([record[target_attr] for record in data]).T
    x = np.matrix([[val*record[target_attr] for ind, val in enumerate(record) if ind != target_attr] for record in data])

    w = np.matrix([1.0]*x.shape[1]).T

    M = [row for row in x if row*w < 0]
    count = 1
    print "Iteration %i - Total mistakes: %i" % (count, len(M))

    while len(M) > 0:
        for i in range(len(w)):
            w[i] = w[i] + learn_rate*(sum([row[0,i] for row in M]))

        M = [row for row in x if row*w < 0]
        count += 1
        print "Iteration %i - Total mistakes: %i" % (count, len(M))

    print "Classifier Weights: ", w.T
    return w


def apply_perceptron(percept, record):
    return (record[1:]*percept[1:])[0,0]


def mean_squared_error_perceptron(data, percept, target_attr):
    return np.mean(map(lambda rec: (apply_perceptron(percept, [val for ind, val in enumerate(rec) if ind != target_attr]) - rec[target_attr])**2, data))

# percept = perceptron(perceptron_data, 5, 1.0)
# c = percept[0,0]
# percept = -percept/c
# percept[0] = c
# print "Normalized with threshold: ", percept[1:].T
# print "Perceptron Training Error LR: %f" % mean_squared_error_perceptron(perceptron_data, percept, 5)

autoencoder_data = [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]]


def sigmoid(x):
    return 1/(1 + math.exp(-x))


def sigmoid_single_layer_neural_network(hidden_size, indata, outdata, learn_rate, iterations):
    input_size = len(indata[0])
    output_size = len(outdata[0])
    w_in_to_hidden = np.random.rand(input_size, hidden_size)
    w_hidden_to_out = np.random.rand(hidden_size, output_size)

    bias_hidden = np.zeros(shape=hidden_size)
    bias_output = np.zeros(shape=output_size)

    for i in range(iterations):
        for ind, record in enumerate(indata):
            i_hidden = np.array(np.matrix(record) * w_in_to_hidden)[0] + bias_hidden
            o_hidden = np.array(map(sigmoid, i_hidden))

            i_output = np.array(np.matrix(o_hidden) * w_hidden_to_out)[0] + bias_output
            o_output = np.array(map(sigmoid, i_output))

            errs_output = np.array([o_output[j]*(1 - o_output[j])*(outdata[ind][j]) for j in range(output_size)])
            errs_hidden = np.array([o_hidden[j]*(1 - o_hidden[j])*(np.dot(errs_output,w_hidden_to_out[j,:])) for j in range(hidden_size)])

            del_w_in_to_hidden = np.zeros(shape=(input_size, hidden_size))
            for i in range(input_size):
                for j in range(hidden_size):
                    del_w_in_to_hidden[i,j] = learn_rate * errs_hidden[j] * record[i]

            del_w_hidden_to_out = np.zeros(shape=(hidden_size, output_size))
            for i in range(hidden_size):
                for j in range(output_size):
                    del_w_hidden_to_out[i,j] = learn_rate * errs_output[j] * o_hidden[i]

            w_in_to_hidden += del_w_in_to_hidden
            w_hidden_to_out += del_w_hidden_to_out

            bias_hidden += learn_rate * errs_hidden
            bias_output += learn_rate * errs_output

    return (w_in_to_hidden, w_hidden_to_out, bias_hidden, bias_output)

def apply_neural_network(nn, data):
    w_in_to_hidden, w_hidden_to_out, bias_hidden, bias_output = nn

    i_hidden = np.array(np.matrix(data) * w_in_to_hidden)[0] + bias_hidden
    o_hidden = np.array(map(sigmoid, i_hidden))

    i_output = np.array(np.matrix(o_hidden) * w_hidden_to_out)[0] + bias_output
    o_output = np.array(map(sigmoid, i_output))

    return o_output

# autoencoder_nn = sigmoid_single_layer_neural_network(3, autoencoder_data, autoencoder_data, 1, 1000)
# print autoencoder_nn
# print apply_neural_network(autoencoder_nn, autoencoder_data[2])

