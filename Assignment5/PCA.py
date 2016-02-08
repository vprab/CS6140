__author__ = 'Vishant'
import numpy as np
import math
import scipy.stats
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB


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


#### GAUSSIAN NAIVE BAYES ####


def gaussian_naive_bayes(data, target_attr):
    m, n = np.shape(data)
    y = data[:,target_attr]
    x = data[:, [i for i in range(n) if i != target_attr]]
    x_zeros = data[[i for i in range(m) if y[i] == -1], :][:, [j for j in range(n) if j != target_attr]]
    x_ones = data[[i for i in range(m) if y[i] == 1], :][:, [j for j in range(n) if j != target_attr]]

    mu_zeros = np.squeeze(np.asarray(np.mean(x_zeros, axis=0)))
    mu_ones = np.squeeze(np.asarray(np.mean(x_ones, axis=0)))

    var_zeros = np.squeeze(np.asarray(np.var(x_zeros, axis=0)))
    var_ones = np.squeeze(np.asarray(np.var(x_ones, axis=0)))

    p_zeros = []
    for mu, var in zip(mu_zeros, var_zeros):
        p_zeros.append((mu, var))

    p_ones = []
    for mu, var in zip(mu_ones, var_ones):
        p_ones.append((mu, var))

    return p_zeros, p_ones


def apply_gaussian_naive_bayes(gaussian_model, record):
    p_zeros, p_ones = gaussian_model

    p_zero = p_one = 1
    for zeros, ones, x in zip(p_zeros, p_ones, record):
        p_zero *= scipy.stats.norm(zeros[0], zeros[1]).pdf(x)
        p_one *= scipy.stats.norm(zeros[0], zeros[1]).pdf(x)

    if p_one >= p_zero:
        return 1
    else:
        return -1


if __name__ == "__main__":
    gnb = GaussianNB()
    gnb.fit(spambase_polluted_train_data, spambase_polluted_train_labels)

    train_pred = np.matrix(gnb.predict(spambase_polluted_train_data)).T
    print "Gaussian Naive Bayes Training Accuracy: %f" % (np.sum(spambase_polluted_train_labels == train_pred).astype(np.float)/np.shape(spambase_polluted_train_labels)[0])

    test_pred = np.matrix(gnb.predict(spambase_polluted_test_data)).T
    print "Gaussian Naive Bayes Testing Accuracy: %f" % (np.sum(spambase_polluted_test_labels == test_pred).astype(np.float)/np.shape(spambase_polluted_test_labels)[0])

    pca = PCA(n_components=100)
    pca.fit(spambase_polluted_train_data)

    train_transform = pca.transform(spambase_polluted_train_data)
    test_transform = pca.transform(spambase_polluted_test_data)

    gnb = GaussianNB()
    gnb.fit(train_transform, spambase_polluted_train_labels)

    train_pred = np.matrix(gnb.predict(train_transform)).T
    print "PCA Gaussian Bayes Training Accuracy: %f" % (np.sum(spambase_polluted_train_labels == train_pred).astype(np.float)/np.shape(spambase_polluted_train_labels)[0])

    test_pred = np.matrix(gnb.predict(test_transform)).T
    print "PCA Gaussian Naive Bayes Testing Accuracy: %f" % (np.sum(spambase_polluted_test_labels == test_pred).astype(np.float)/np.shape(spambase_polluted_test_labels)[0])



