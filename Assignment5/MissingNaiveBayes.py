__author__ = 'Vishant'
import numpy as np
from sklearn.naive_bayes import GaussianNB


#### LOADING DATA ####


with open("20_percent_missing_train.txt") as spambase:
    spambase_train_data = [line.split(',') for line in spambase]

spambase_train_data = np.array(spambase_train_data).astype(np.float)

with open("20_percent_missing_test.txt") as spambase:
    spambase_test_data = [line.split(',') for line in spambase]

spambase_test_data = np.array(spambase_test_data).astype(np.float)


#### BERNOULLI NAIVE BAYES ####


def bernoulli_naive_bayes(data, target_attr):
    m, n = np.shape(data)
    y = data[:,target_attr]
    x = data[:, [i for i in range(n) if i != target_attr]]
    x_zeros = data[[i for i in range(m) if y[i] == 0], :][:, [j for j in range(n) if j != target_attr]]
    x_ones = data[[i for i in range(m) if y[i] == 1], :][:, [j for j in range(n) if j != target_attr]]

    mu_all = np.array([np.mean(x[:,i][~np.isnan(x[:,i])]) for i in range(n-1)])

    spam_above_means = np.array([np.sum((x_ones[:,i][~np.isnan(x_ones[:,i])] > mu_all[i]).astype(np.float))/np.shape(x_ones[:,i][~np.isnan(x_ones[:,i])])[0] for i in range(n-1)])
    spam_below_means = np.array([np.sum((x_ones[:,i][~np.isnan(x_ones[:,i])] <= mu_all[i]).astype(np.float))/np.shape(x_ones[:,i][~np.isnan(x_ones[:,i])])[0] for i in range(n-1)])
    nonspam_above_means = np.array([np.sum((x_zeros[:,i][~np.isnan(x_zeros[:,i])] > mu_all[i]).astype(np.float))/np.shape(x_zeros[:,i][~np.isnan(x_zeros[:,i])])[0] for i in range(n-1)])
    nonspam_below_means = np.array([np.sum((x_zeros[:,i][~np.isnan(x_zeros[:,i])] <= mu_all[i]).astype(np.float))/np.shape(x_zeros[:,i][~np.isnan(x_zeros[:,i])])[0] for i in range(n-1)])

    # spam_above_means = np.sum((x_ones > mu_all).astype(np.float), axis=0)/(np.shape(x_ones)[0])
    # spam_below_means = np.sum((x_ones <= mu_all).astype(np.float), axis=0)/(np.shape(x_ones)[0])
    # nonspam_above_means = np.sum((x_zeros > mu_all).astype(np.float), axis=0)/(np.shape(x_zeros)[0])
    # nonspam_below_means = np.sum((x_zeros <= mu_all).astype(np.float), axis=0)/(np.shape(x_zeros)[0])

    return mu_all, spam_above_means, spam_below_means, nonspam_above_means, nonspam_below_means


def apply_bernoulli_naive_bayes(bernoulli_model, record):
    mu_all, spam_above_means, spam_below_means, nonspam_above_means, nonspam_below_means = bernoulli_model

    p_spam = 1
    p_nonspam = 1
    # comps = np.squeeze(np.asarray(record > mu_all))

    for ind, val in enumerate(np.asarray(record)):
        if val is not np.nan:
            if val > mu_all[ind]:
                p_spam *= spam_above_means[ind]
                p_nonspam *= nonspam_above_means[ind]
            else:
                p_spam *= spam_below_means[ind]
                p_nonspam *= nonspam_below_means[ind]

    if p_spam >= p_nonspam:
        return 1
    else:
        return 0

bernoulli_nb_model = bernoulli_naive_bayes(spambase_train_data, 57)
test_acc = 0.0
for record in spambase_test_data:
    y_obs = record[57]
    x_obs = np.squeeze(np.array(np.matrix(record[:57])))

    pred_value = apply_bernoulli_naive_bayes(bernoulli_nb_model, x_obs)
    if pred_value == y_obs:
        test_acc += 1

print "Naive Bayes Missing Value Accuracy: %f" % (test_acc/(np.shape(spambase_test_data)[0]))