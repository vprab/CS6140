__author__ = 'Vishant'
import numpy as np
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
import math


#### LOADING DATA ####


with open("spam_polluted/train_feature.txt") as spam_polluted_features, open("spam_polluted/train_label.txt") as spam_polluted_labels:
    spambase_polluted_train_data = [f.split() for f in spam_polluted_features]
    spambase_polluted_train_labels = [l.split() for l in spam_polluted_labels]

spambase_polluted_train_data = np.array(spambase_polluted_train_data).astype(np.float)
spambase_polluted_train_labels = np.array(spambase_polluted_train_labels).astype(np.float)

with open("spam_polluted/test_feature.txt") as spam_polluted_features, open("spam_polluted/test_label.txt") as spam_polluted_labels:
    spambase_polluted_test_data = [f.split() for f in spam_polluted_features]
    spambase_polluted_test_labels = [l.split() for l in spam_polluted_labels]

spambase_polluted_test_data = np.array(spambase_polluted_test_data).astype(np.float)
spambase_polluted_test_labels = np.array(spambase_polluted_test_labels).astype(np.float)


#### GRADIENT DESCENT LOGISTIC REGRESSION ####


def sigmoid(x):
    return 1/(1 + math.exp(-x))


def logistic_regression_grad_desc(data, labels, learn_rate, iterations):
    y = labels
    x = data

    m, n = np.shape(x)
    theta = np.zeros(shape=n)

    for i in range(0, iterations):
        for t in range(m):
            for j in range(len(theta)):
                theta[j] = theta[j] - (learn_rate * (sigmoid(np.dot(theta, x[t])) - y[t]) * x[t,j])

    return theta


def apply_log_reg(beta, record):
    np_record = np.array(record)
    return sigmoid(np.dot(np_record, beta))

if __name__ == "__main__":
    lr = LogisticRegression()
    lr.fit(spambase_polluted_train_data, spambase_polluted_train_labels)

    train_pred = np.matrix(lr.predict(spambase_polluted_train_data)).T
    print "Logistic Regression Training Accuracy: %f" % (np.sum(spambase_polluted_train_labels == train_pred).astype(np.float)/np.shape(spambase_polluted_train_labels)[0])

    test_pred = np.matrix(lr.predict(spambase_polluted_test_data)).T
    print "Logistic Regression Testing Accuracy: %f" % (np.sum(spambase_polluted_test_labels == test_pred).astype(np.float)/np.shape(spambase_polluted_test_labels)[0])

    # lasso = Lasso()
    # lasso.fit(spambase_polluted_train_data, spambase_polluted_train_labels)
    #
    # train_pred = np.matrix(lasso.predict(spambase_polluted_train_data)).T
    # print "Lasso Training Accuracy: %f" % (np.sum(spambase_polluted_train_labels == (train_pred > 0.5)).astype(np.float)/np.shape(spambase_polluted_train_labels)[0])
    #
    # test_pred = np.matrix(lasso.predict(spambase_polluted_test_data)).T
    # print "Lasso Testing Accuracy: %f" % (np.sum(spambase_polluted_test_labels == (test_pred > 0.5)).astype(np.float)/np.shape(spambase_polluted_test_labels)[0])

    ridge = Ridge()
    ridge.fit(spambase_polluted_train_data, spambase_polluted_train_labels)

    train_pred = np.matrix(ridge.predict(spambase_polluted_train_data))
    print "Ridge Training Accuracy: %f" % (np.sum(spambase_polluted_train_labels == (train_pred > 0.5)).astype(np.float)/np.shape(spambase_polluted_train_labels)[0])

    test_pred = np.matrix(ridge.predict(spambase_polluted_test_data))
    print "Ridge Testing Accuracy: %f" % (np.sum(spambase_polluted_test_labels == (test_pred > 0.5)).astype(np.float)/np.shape(spambase_polluted_test_labels)[0])
