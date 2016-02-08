__author__ = 'Vishant'
import numpy as np
import random
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

#### LOADING DATA ####


with open("spambase.data") as spambase:
    spambase_data = [line.split(',') for line in spambase]

spambase_data = np.array(spambase_data).astype(np.float)
for i in range(np.shape(spambase_data)[0]):
    if spambase_data[i][-1] == 0:
        spambase_data[i][-1] = -1

K = 10
split_spambase = np.array([spambase_data[i::K] for i in range(K)])

with open("spam_polluted/train_feature.txt") as spam_polluted_features, open("spam_polluted/train_label.txt") as spam_polluted_labels:
    spambase_polluted_train_data = [f.split() + l.split() for f, l in zip(spam_polluted_features, spam_polluted_labels)]

spambase_polluted_train_data = np.array(spambase_polluted_train_data).astype(np.float)
for i in range(np.shape(spambase_polluted_train_data)[0]):
    if spambase_polluted_train_data[i][-1] == 0:
        spambase_polluted_train_data[i][-1] = -1

with open("spam_polluted/test_feature.txt") as spam_polluted_features, open("spam_polluted/test_label.txt") as spam_polluted_labels:
    spambase_polluted_test_data = [f.split() + l.split() for f, l in zip(spam_polluted_features, spam_polluted_labels)]

spambase_polluted_test_data = np.array(spambase_polluted_test_data).astype(np.float)
for i in range(np.shape(spambase_polluted_test_data)[0]):
    if spambase_polluted_test_data[i][-1] == 0:
        spambase_polluted_test_data[i][-1] = -1


#### DECISION STUMP ADABOOST ####


def optimal_decision_stump(x, y, distribution=None):
    m, n = np.shape(x)
    ans_attr = None
    ans_val = None
    opt_error = 0

    if distribution is None:
        distribution = np.array([1.0/m]*m)

    eval_dict = {i:{} for i in range(n)}

    for ind, record in enumerate(x):
        for attr in range(n):
            val = record[attr]
            dist_value = distribution[ind]

            if val not in eval_dict[attr].keys():
                eval_dict[attr][val] = [0.0, 0.0]

            if y[ind] == -1 or y[ind] == 0:
                eval_dict[attr][val][0] += dist_value
            else:
                eval_dict[attr][val][1] += dist_value

    for attr in range(n):
        dict_attr = eval_dict[attr]

        for val in dict_attr.keys():
            error = 0.0

            for comp_val in dict_attr.keys():
                dict_attr_val = dict_attr[comp_val]

                if comp_val < val:
                    error += dict_attr_val[1]
                else:
                    error += dict_attr_val[0]

            true_error = np.abs(1/2.0 - error)
            if true_error > opt_error:
                opt_error = true_error
                ans_attr = attr
                ans_val = val

    return ans_attr, ans_val


def random_decision_stump(x, y, distribution):
    m, n = np.shape(x)
    ans_attr = random.choice(range(n))
    ans_val = random.choice(list(set([record[ans_attr] for record in x])))

    return ans_attr, ans_val


def apply_decision_stump(record, stump):
    attr, val = stump

    if record[attr] >= val:
        return 1
    else:
        return -1


def adaboost_decision_stump(data, target_attr, weak_learner, iterations):
    m, n = np.shape(data)
    y = data[:,target_attr]
    x = data[:, [i for i in range(n) if i != target_attr]]
    distribution = np.array([1.0/m]*m)

    ans = []
    for t in range(0, iterations):
        h = weak_learner(x, y, distribution)

        epsilon = 0.0
        for ind, record in enumerate(x):
            if apply_decision_stump(record, h) != y[ind]:
                epsilon += distribution[ind]

        alpha = 1/2.0 * np.log(max((1-epsilon)/max(epsilon, 1e-300), 1e-300))

        for ind, record in enumerate(x):
            distribution[ind] *= np.exp(-alpha*y[ind]*apply_decision_stump(record, h))

        distribution /= np.sum(distribution, axis=0)
        ans.append((alpha, h))

    return ans


def apply_adaboost_decision_stump(record, ada_stump):
    ans = 0

    for alpha, stump in ada_stump:
        ans += alpha*apply_decision_stump(record, stump)

    return ans


def decision_stump_error(data, target_attr, stump):
    m, n = np.shape(data)
    x = data[:, [i for i in range(n) if i != target_attr]]
    y = data[:,target_attr]

    error = 0.0
    for ind, record in enumerate(x):
        if apply_decision_stump(record, stump) != y[ind]:
            error += 1.0/m

    return error


def adaboost_decision_stump_error(data, target_attr, ada_stump):
    m, n = np.shape(data)
    x = data[:, [i for i in range(n) if i != target_attr]]
    y = data[:,target_attr]

    error = 0.0
    for ind, record in enumerate(x):
        if np.sign(apply_adaboost_decision_stump(record, ada_stump)) != y[ind]:
            error += 1.0/m

    return error


if __name__ == "__main__":
    testing_data = split_spambase[0]
    training_data = np.array([item for sublist in [x for ind, x in enumerate(split_spambase) if ind != 0] for item in sublist])

    ada_stumps = adaboost_decision_stump(training_data, 57, random_decision_stump, 1000)

    total_margin = 0.0
    for i in range(len(training_data)):
        for s in ada_stumps:
            total_margin += training_data[i][57]*s[0]*apply_decision_stump(training_data[i], s[1])

    margins = [0.0]*57
    for m in range(57):
        ans = 0.0

        f_stumps = [s for s in ada_stumps if s[1][0] == m]

        for i in range(len(training_data)):
            for s in f_stumps:
                ans += training_data[i][57]*s[0]*apply_decision_stump(training_data[i], s[1])

        ans /= total_margin
        margins[m] = ans

    print sorted(range(len(margins)), key=lambda i:-margins[i])
    #
    # bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=300)
    # bdt.fit(training_data[:,:-1], training_data[:,-1])
    #
    # print sorted(range(len(bdt.feature_importances_)), key=lambda i: -bdt.feature_importances_[i])

    # print "-----------------------------------------"
    #
    # m_train, n_train = np.shape(spambase_polluted_train_data)
    # m_test, n_test = np.shape(spambase_polluted_test_data)
    # ada_stumps = adaboost_decision_stump(spambase_polluted_train_data, n_train-1, optimal_decision_stump, 50)
    # print adaboost_decision_stump_error(spambase_polluted_train_data, n_train-1, ada_stumps)
    # print adaboost_decision_stump_error(spambase_polluted_test_data, n_test-1, ada_stumps)

    # bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=50)
    # bdt.fit(spambase_polluted_train_data[:,:-1], spambase_polluted_train_data[:,-1])
    #
    # train_pred = bdt.predict(spambase_polluted_train_data[:,:-1])
    # train_labels = spambase_polluted_train_data[:,-1]
    # print "AdaBoost Polluted Training Accuracy: %f" % (np.sum(train_pred == train_labels).astype(np.float)/np.shape(train_labels)[0])
    #
    # test_pred = bdt.predict(spambase_polluted_test_data[:,:-1])
    # test_labels = spambase_polluted_test_data[:,-1]
    # print "AdaBoost Polluted Testing Accuracy: %f" % (np.sum(test_pred == test_labels).astype(np.float)/np.shape(test_labels)[0])
