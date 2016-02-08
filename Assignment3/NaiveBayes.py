from __future__ import division
__author__ = 'Vishant'
import numpy as np
import math
from tabulate import tabulate
import matplotlib.pyplot as plt

#### LOADING DATA ####

with open("spambase.data") as spambase:
    spambase_data = [line.split(',') for line in spambase]

spambase_data = np.array(spambase_data).astype(np.float)
spambase_data_norm = np.copy(spambase_data)

K = 10
split_spambase = np.array([spambase_data[i::K] for i in range(K)])

#### BERNOULLI NAIVE BAYES ####


def bernoulli_naive_bayes(data, target_attr):
    m, n = np.shape(data)
    y = data[:,target_attr]
    x = data[:, [i for i in range(n) if i != target_attr]]
    x_zeros = data[[i for i in range(m) if y[i] == 0], :][:, [j for j in range(n) if j != target_attr]]
    x_ones = data[[i for i in range(m) if y[i] == 1], :][:, [j for j in range(n) if j != target_attr]]

    mu_all = np.mean(x, axis=0)

    spam_above_means = np.sum((x_ones > mu_all).astype(np.float), axis=0)/(np.shape(x_ones)[0])
    spam_below_means = np.sum((x_ones <= mu_all).astype(np.float), axis=0)/(np.shape(x_ones)[0])
    nonspam_above_means = np.sum((x_zeros > mu_all).astype(np.float), axis=0)/(np.shape(x_zeros)[0])
    nonspam_below_means = np.sum((x_zeros <= mu_all).astype(np.float), axis=0)/(np.shape(x_zeros)[0])

    return mu_all, spam_above_means, spam_below_means, nonspam_above_means, nonspam_below_means


def apply_bernoulli_naive_bayes(bernoulli_model, record):
    mu_all, spam_above_means, spam_below_means, nonspam_above_means, nonspam_below_means = bernoulli_model

    p_spam = 1
    p_nonspam = 1
    comps = np.squeeze(np.asarray(record > mu_all))

    for ind, val in enumerate(comps):
        if val:
            p_spam *= spam_above_means[ind]
            p_nonspam *= nonspam_above_means[ind]
        else:
            p_spam *= spam_below_means[ind]
            p_nonspam *= nonspam_below_means[ind]

    if p_spam >= p_nonspam:
        return 1
    else:
        return 0

#### BERNOULLI NAIVE BAYES - TESTS ####

bernoulli_accuracies = []
bernoulli_fp_rates = []
bernoulli_fn_rates = []
bernoulli_error_rates = []
fold_names = ["Fold %i" % count for count in range(1, 11)]
fold_names.append("Average")

for i in range(K):
    testing_data = split_spambase[i]
    training_data = np.array([item for sublist in [x for ind, x in enumerate(split_spambase) if ind != i] for item in sublist])

    bernoulli_nb_model = bernoulli_naive_bayes(training_data, 57)

    test_acc = 0.0
    false_positives = 0.0
    false_negatives = 0.0
    true_positives = 0.0
    true_negatives = 0.0
    errors = 0.0
    for record in testing_data:
        y_obs = record[57]
        x_obs = np.squeeze(np.array(np.matrix(record[:57])))

        pred_value = apply_bernoulli_naive_bayes(bernoulli_nb_model, x_obs)
        if pred_value == y_obs:
            test_acc += 1

        if pred_value == 1 and y_obs == 1:
            true_positives += 1
        elif pred_value == 1 and y_obs == 0:
            false_positives += 1
            errors += 1
        elif pred_value == 0 and y_obs == 1:
            false_negatives += 1
            errors += 1
        elif pred_value == 0 and y_obs == 0:
            true_negatives += 1

    bernoulli_fp_rates.append(false_positives/(false_positives + true_negatives))
    bernoulli_fn_rates.append(false_negatives/(false_negatives + true_positives))
    bernoulli_error_rates.append(errors/(np.shape(testing_data)[0]))
    bernoulli_accuracies.append(test_acc/(np.shape(testing_data)[0]))

bernoulli_fp_rates.append(np.mean(bernoulli_fp_rates))
bernoulli_fn_rates.append(np.mean(bernoulli_fn_rates))
bernoulli_error_rates.append(np.mean(bernoulli_error_rates))
bernoulli_accuracies.append(np.mean(bernoulli_accuracies))
print tabulate(zip(fold_names, bernoulli_fp_rates, bernoulli_fn_rates, bernoulli_error_rates, bernoulli_accuracies), headers=['Bernoulli Naive Bayes', 'FP Rate', 'FN Rate', 'Error Rate', 'Accuracy'], tablefmt='orgtbl')
print "\n"

#### GAUSSIAN NAIVE BAYES ####


def normal(x, mu, var):
    var_e = max(var, 0.001)
    return 1/(math.sqrt(2*np.pi*var_e)) * math.exp(-((x - mu) ** 2)/(2*var_e))


def gaussian_naive_bayes(data, target_attr):
    m, n = np.shape(data)
    y = data[:,target_attr]
    x = data[:, [i for i in range(n) if i != target_attr]]
    x_zeros = data[[i for i in range(m) if y[i] == 0], :][:, [j for j in range(n) if j != target_attr]]
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
        p_zero *= normal(x, zeros[0], zeros[1])
        p_one *= normal(x, ones[0], ones[1])

    if p_one >= p_zero:
        return 1
    else:
        return 0

#### GAUSSIAN NAIVE BAYES - TESTS ####

gaussian_accuracies = []
gaussian_fp_rates = []
gaussian_fn_rates = []
gaussian_error_rates = []

for i in range(K):
    testing_data = split_spambase[i]
    training_data = np.array([item for sublist in [x for ind, x in enumerate(split_spambase) if ind != i] for item in sublist])

    gaussian_nb_model = gaussian_naive_bayes(training_data, 57)

    test_acc = 0.0
    false_positives = 0.0
    false_negatives = 0.0
    true_positives = 0.0
    true_negatives = 0.0
    errors = 0.0
    for record in testing_data:
        y_obs = record[57]
        x_obs = np.squeeze(np.array(np.matrix(record[:57])))

        pred_value = apply_gaussian_naive_bayes(gaussian_nb_model, x_obs)
        if pred_value == y_obs:
            test_acc += 1

        if pred_value == 1 and y_obs == 1:
            true_positives += 1
        elif pred_value == 1 and y_obs == 0:
            false_positives += 1
            errors += 1
        elif pred_value == 0 and y_obs == 1:
            false_negatives += 1
            errors += 1
        elif pred_value == 0 and y_obs == 0:
            true_negatives += 1

    gaussian_fp_rates.append(false_positives/(false_positives + true_negatives))
    gaussian_fn_rates.append(false_negatives/(false_negatives + true_positives))
    gaussian_error_rates.append(errors/(np.shape(testing_data)[0]))
    gaussian_accuracies.append(test_acc/(np.shape(testing_data)[0]))

gaussian_fp_rates.append(np.mean(gaussian_fp_rates))
gaussian_fn_rates.append(np.mean(gaussian_fn_rates))
gaussian_error_rates.append(np.mean(gaussian_error_rates))
gaussian_accuracies.append(np.mean(gaussian_accuracies))
print tabulate(zip(fold_names, gaussian_fp_rates, gaussian_fn_rates, gaussian_error_rates, gaussian_accuracies), headers=['Gaussian Naive Bayes', 'FP Rate', 'FN Rate', 'Error Rate', 'Accuracy'], tablefmt='orgtbl')
print "\n"

#### HISTOGRAM NAIVE BAYES ####


def bin_naive_bayes(data, target_attr):
    m, n = np.shape(data)
    y = data[:,target_attr]
    x = data[:, [i for i in range(n) if i != target_attr]]
    x_zeros = data[[i for i in range(m) if y[i] == 0], :][:, [j for j in range(n) if j != target_attr]]
    x_ones = data[[i for i in range(m) if y[i] == 1], :][:, [j for j in range(n) if j != target_attr]]

    mu_zeros = np.squeeze(np.asarray(np.mean(x_zeros, axis=0)))
    mu_ones = np.squeeze(np.asarray(np.mean(x_ones, axis=0)))
    mu_all = np.squeeze(np.asarray(np.mean(x, axis=0)))

    mins = np.squeeze(np.asarray(np.amin(x, axis=0)))
    maxs = np.squeeze(np.asarray(np.amax(x, axis=0)))

    intervals = zip(mins, maxs, mu_zeros, mu_ones, mu_all)
    intervals = map(lambda x: sorted(x), intervals)

    p_zeros = np.zeros(shape=(n-1, 4))
    for record in x_zeros:
        for ind, val in enumerate(np.squeeze(np.asarray(record))):
            ivals = intervals[ind]

            if ivals[0] <= val <= ivals[1]:
                p_zeros[ind, 0] += 1
            elif ivals[1] < val <= ivals[2]:
                p_zeros[ind, 1] += 1
            elif ivals[2] < val <= ivals[3]:
                p_zeros[ind, 2] += 1
            elif ivals[3] < val <= ivals[4]:
                p_zeros[ind, 3] += 1

    p_zeros /= np.shape(x_zeros)[0]

    p_ones = np.zeros(shape=(n-1, 4))
    for record in x_ones:
        for ind, val in enumerate(np.squeeze(np.asarray(record))):
            ivals = intervals[ind]

            if ivals[0] <= val <= ivals[1]:
                p_ones[ind, 0] += 1
            elif ivals[1] < val <= ivals[2]:
                p_ones[ind, 1] += 1
            elif ivals[2] < val <= ivals[3]:
                p_ones[ind, 2] += 1
            elif ivals[3] < val <= ivals[4]:
                p_ones[ind, 3] += 1

    p_ones /= np.shape(x_ones)[0]

    return intervals, p_zeros, p_ones


def apply_bin_naive_bayes(bin_model, record):
    intervals, p_zeros, p_ones = bin_model

    p_spam = 1
    p_nonspam = 1

    for ind, val in enumerate(record):
        ivals = intervals[ind]

        if ivals[0] <= val <= ivals[1]:
            p_nonspam *= p_zeros[ind, 0]
            p_spam *= p_ones[ind, 0]
        elif ivals[1] < val <= ivals[2]:
            p_nonspam *= p_zeros[ind, 1]
            p_spam *= p_ones[ind, 1]
        elif ivals[2] < val <= ivals[3]:
            p_nonspam *= p_zeros[ind, 2]
            p_spam *= p_ones[ind, 2]
        elif ivals[3] < val <= ivals[4]:
            p_nonspam *= p_zeros[ind, 3]
            p_spam *= p_ones[ind, 3]

    if p_spam >= p_nonspam:
        return 1
    else:
        return 0

#### BIN NAIVE BAYES - TESTS ####

bin_accuracies = []
bin_fp_rates = []
bin_fn_rates = []
bin_error_rates = []
fold_names = ["Fold %i" % count for count in range(1, 11)]
fold_names.append("Average")

for i in range(K):
    testing_data = split_spambase[i]
    training_data = np.array([item for sublist in [x for ind, x in enumerate(split_spambase) if ind != i] for item in sublist])

    bin_nb_model = bin_naive_bayes(training_data, 57)

    test_acc = 0.0
    false_positives = 0.0
    false_negatives = 0.0
    true_positives = 0.0
    true_negatives = 0.0
    errors = 0.0
    for record in testing_data:
        y_obs = record[57]
        x_obs = np.squeeze(np.array(np.matrix(record[:57])))

        pred_value = apply_bin_naive_bayes(bin_nb_model, x_obs)
        if pred_value == y_obs:
            test_acc += 1

        if pred_value == 1 and y_obs == 1:
            true_positives += 1
        elif pred_value == 1 and y_obs == 0:
            false_positives += 1
            errors += 1
        elif pred_value == 0 and y_obs == 1:
            false_negatives += 1
            errors += 1
        elif pred_value == 0 and y_obs == 0:
            true_negatives += 1

    bin_fp_rates.append(false_positives/(false_positives + true_negatives))
    bin_fn_rates.append(false_negatives/(false_negatives + true_positives))
    bin_error_rates.append(errors/(np.shape(testing_data)[0]))
    bin_accuracies.append(test_acc/(np.shape(testing_data)[0]))

bin_fp_rates.append(np.mean(bin_fp_rates))
bin_fn_rates.append(np.mean(bin_fn_rates))
bin_error_rates.append(np.mean(bin_error_rates))
bin_accuracies.append(np.mean(bin_accuracies))
print tabulate(zip(fold_names, bin_fp_rates, bin_fn_rates, bin_error_rates, bin_accuracies), headers=['Histogram Naive Bayes', 'FP Rate', 'FN Rate', 'Error Rate', 'Accuracy'], tablefmt='orgtbl')
print "\n"

#### ROC CURVES ####

testing_data = split_spambase[0]
training_data = np.array([item for sublist in [x for ind, x in enumerate(split_spambase) if ind != 0] for item in sublist])
bernoulli_nb_model = bernoulli_naive_bayes(training_data, 57)
gaussian_nb_model = gaussian_naive_bayes(training_data, 57)
bin_nb_model = bin_naive_bayes(training_data, 57)
mu_all, spam_above_means, spam_below_means, nonspam_above_means, nonspam_below_means = bernoulli_nb_model
p_nonspam_gaussians, p_spam_gaussians = gaussian_nb_model
intervals, p_nonspam_bins, p_spam_bins = bin_nb_model

bernoulli_log_odds = []
gaussian_log_odds = []
bin_log_odds = []
obs_pos = obs_neg = 0.0
for record in testing_data:
    y_obs = record[57]
    x_obs = np.squeeze(np.array(np.matrix(record[:57])))

    p_spam_bernoulli = 1
    p_nonspam_bernoulli = 1
    comps = np.squeeze(np.asarray(x_obs > mu_all))

    for ind, val in enumerate(comps):
        if val:
            p_spam_bernoulli *= spam_above_means[ind]
            p_nonspam_bernoulli *= nonspam_above_means[ind]
        else:
            p_spam_bernoulli *= spam_below_means[ind]
            p_nonspam_bernoulli *= nonspam_below_means[ind]

    bernoulli_log_odds.append((math.log(p_spam_bernoulli/p_nonspam_bernoulli), y_obs))

    p_nonspam_gaussian = p_spam_gaussian = 1
    for zeros, ones, x in zip(p_nonspam_gaussians, p_spam_gaussians, x_obs):
        p_nonspam_gaussian *= max(normal(x, zeros[0], zeros[1]), 0.001)
        p_spam_gaussian *= max(normal(x, ones[0], ones[1]), 0.001)

    gaussian_log_odds.append((math.log(p_spam_gaussian/p_nonspam_gaussian), y_obs))

    p_spam_bin = 1
    p_nonspam_bin = 1

    for ind, val in enumerate(x_obs):
        ivals = intervals[ind]

        if ivals[0] <= val <= ivals[1]:
            p_nonspam_bin *= max(p_nonspam_bins[ind, 0], 0.01)
            p_spam_bin *= max(p_spam_bins[ind, 0], 0.01)
        elif ivals[1] < val <= ivals[2]:
            p_nonspam_bin *= max(p_nonspam_bins[ind, 1], 0.01)
            p_spam_bin *= max(p_spam_bins[ind, 1], 0.01)
        elif ivals[2] < val <= ivals[3]:
            p_nonspam_bin *= max(p_nonspam_bins[ind, 2], 0.01)
            p_spam_bin *= max(p_spam_bins[ind, 2], 0.01)
        elif ivals[3] < val <= ivals[4]:
            p_nonspam_bin *= max(p_nonspam_bins[ind, 3], 0.01)
            p_spam_bin *= max(p_spam_bins[ind, 3], 0.01)

    bin_log_odds.append((math.log(p_spam_bin/p_nonspam_bin), y_obs))

    if y_obs == 1:
        obs_pos += 1

    if y_obs == 0:
        obs_neg += 1


bernoulli_log_odds = sorted(bernoulli_log_odds, key=lambda x: -x[0])
tps = []
fps = []

for i in range(len(bernoulli_log_odds)):
    predicted_pos = bernoulli_log_odds[:i]

    true_pos = sum(map(lambda x: x[1], predicted_pos))
    false_pos = len(predicted_pos) - true_pos

    tps.append(true_pos/obs_pos)
    fps.append(false_pos/obs_neg)

plt.plot(fps, tps, linewidth=1, label="Bernoulli")

auc = np.trapz(tps, x=fps)
print "Bernoulli Bayes AUC: %f" % auc

gaussian_log_odds = sorted(gaussian_log_odds, key=lambda x: -x[0])
tps = []
fps = []

for i in range(len(gaussian_log_odds)):
    predicted_pos = gaussian_log_odds[:i]

    true_pos = sum(map(lambda x: x[1], predicted_pos))
    false_pos = len(predicted_pos) - true_pos

    tps.append(true_pos/obs_pos)
    fps.append(false_pos/obs_neg)

plt.plot(fps, tps, linewidth=1, label="Gaussian")

auc = np.trapz(tps, x=fps)
print "Gaussian Bayes AUC: %f" % auc

bin_log_odds = sorted(bin_log_odds, key=lambda x: -x[0])
tps = []
fps = []

for i in range(len(bin_log_odds)):
    predicted_pos = bin_log_odds[:i]

    true_pos = sum(map(lambda x: x[1], predicted_pos))
    false_pos = len(predicted_pos) - true_pos

    tps.append(true_pos/obs_pos)
    fps.append(false_pos/obs_neg)

plt.plot(fps, tps, linewidth=1, label="Histogram")
plt.legend()

auc = np.trapz(tps, x=fps)
print "Histogram Bayes AUC: %f" % auc

plt.show()

