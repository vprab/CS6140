__author__ = 'Vishant'
import numpy as np

### LOADING ALL DATA

def add_ones_column(data):
    return [[1] + row for row in data]

with open("housing_train.txt") as housing_train:
    housing_train_data = [line.split() for line in housing_train]

housing_train_data = [[float(x) for x in y] for y in housing_train_data]
housing_train_data = add_ones_column(housing_train_data)

with open("housing_test.txt") as housing_test:
    housing_test_data = [line.split() for line in housing_test]

housing_test_data = [[float(x) for x in y] for y in housing_test_data]
housing_test_data = add_ones_column(housing_test_data)

with open("perceptronData.txt") as perceptron:
    perceptron_data = [line.split() for line in perceptron]

perceptron_data = [[float(x) for x in y] for y in perceptron_data]
perceptron_data = add_ones_column(perceptron_data)

with open("spambase.data") as spambase:
    spambase_data = [line.split(',') for line in spambase]

K = 10
spambase_data = [[float(x) for x in y] for y in spambase_data]
spambase_data = add_ones_column(spambase_data)
split_spambase = [spambase_data[i::K] for i in range(K)]

def linear_regression_ridge(data, target_attr, reg):
    """

    :param data: A dataset
    :param target_attr: The column number representing the target attribute
    :return: A linear regression vector of coefficients
    """
    y = np.matrix([record[target_attr] for record in data])
    x = np.matrix([[val for ind, val in enumerate(record) if ind != target_attr] for record in data])

    b = (x.T * x + reg*np.identity(x.shape[1])).I * x.T * y.T

    return b

def linear_regression_grad_desc(data, target_attr, reg):


def apply_lin_reg(beta, record):
    """

    :param beta: A linear regression vector of coefficients
    :param record: A record of data
    :return: The result of applying the given vector to the record
    """
    np_record = np.matrix(record)
    return np_record*beta


def mean_squared_error_lr(data, beta, target_attr):
    """

    :param data: A dataset
    :param beta: A linear regression vector of coefficients
    :param target_attr: The column number representing the target attribute
    :return: The MSE between the results of running the linear regression vector on the dataset and the actual target
    values
    """
    return np.mean(map(lambda rec: (apply_lin_reg(beta, [val for ind, val in enumerate(rec) if ind != target_attr]) - rec[target_attr])**2, data))

housing_lin_reg = linear_regression_ridge(housing_train_data, 14, 5)
print "Housing Training Error LR: %f" % mean_squared_error_lr(housing_train_data, housing_lin_reg, 14)
print "Housing Testing Error LR: %f" % mean_squared_error_lr(housing_test_data, housing_lin_reg, 14)

training_errors_lr = []
testing_errors_lr = []

for i in range(K):
    testing_data = split_spambase[i]
    training_data = [item for sublist in [x for ind, x in enumerate(split_spambase) if ind != i] for item in sublist]

    spam_lin_reg = linear_regression_ridge(training_data, 58, 5)
    training_errors_lr.append(mean_squared_error_lr(training_data, spam_lin_reg, 58))
    testing_errors_lr.append(mean_squared_error_lr(testing_data, spam_lin_reg, 58))

print "Spam Average Training Error LR: %f" % np.mean(training_errors_lr)
print "Spam Average Testing Error LR: %f" % np.mean(testing_errors_lr)

