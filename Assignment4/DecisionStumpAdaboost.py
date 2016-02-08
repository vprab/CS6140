__author__ = 'Vishant'
import numpy as np
import math
import random
import matplotlib.pyplot as plt

#### LOADING DATA ####

with open("spambase.data") as spambase:
    spambase_data = [line.split(',') for line in spambase]

spambase_data = np.array(spambase_data).astype(np.float)
for i in range(np.shape(spambase_data)[0]):
    if spambase_data[i][57] == 0:
        spambase_data[i][57] = -1

K = 10
split_spambase = np.array([spambase_data[i::K] for i in range(K)])

#### DECISION STUMP ADABOOST ####


def optimal_decision_stump(x, y, distribution):
    m, n = np.shape(x)
    ans_attr = None
    ans_val = None
    opt_error = 0

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

#### DECISION STUMP ADABOOST - TESTS ####

if __name__ == "__main__":
    # round_errors = []
    # weighted_train_error = []
    # weighted_test_error = []
    #
    # testing_data = split_spambase[0]
    # training_data = np.array([item for sublist in [x for ind, x in enumerate(split_spambase) if ind != 0] for item in sublist])
    #
    # m, n = np.shape(training_data)
    # y = training_data[:,57]
    # x = training_data[:, [i for i in range(n) if i != 57]]
    # distribution = np.array([1.0/m]*m)
    #
    # ans = []
    # for t in range(0, 100):
    #     h = optimal_decision_stump(x, y, distribution)
    #
    #     epsilon = 0.0
    #     local_round_error = 0.0
    #     for ind, record in enumerate(x):
    #         if apply_decision_stump(record, h) != y[ind]:
    #             epsilon += distribution[ind]
    #             local_round_error += distribution[ind]
    #
    #     round_errors.append(local_round_error)
    #     alpha = 1/2.0 * np.log(max((1-epsilon)/epsilon, 1e-300))
    #
    #     for ind, record in enumerate(x):
    #         distribution[ind] *= np.exp(-alpha*y[ind]*apply_decision_stump(record, h))
    #
    #     distribution /= np.sum(distribution, axis=0)
    #     ans.append((alpha, h))
    #
    #     # weighted_train_error.append(adaboost_decision_stump_error(training_data, 57, ans))
    #     # weighted_test_error.append(adaboost_decision_stump_error(testing_data, 57, ans))
    #
    # train_acc = 0.0
    # for record in training_data:
    #     y_obs = record[57]
    #     x_obs = np.matrix(record[:57])
    #
    #     if np.sign(apply_adaboost_decision_stump(record, ans)) == y_obs:
    #         train_acc += 1
    #
    # test_acc = 0.0
    # for record in testing_data:
    #     y_obs = record[57]
    #     x_obs = np.matrix(record[:57])
    #
    #     if np.sign(apply_adaboost_decision_stump(record, ans)) == y_obs:
    #         test_acc += 1
    #
    # train_accuracy = train_acc/(np.shape(training_data)[0])
    # test_accuracy = test_acc/(np.shape(testing_data)[0])
    #
    #
    # print "Training Accuracy: %f" % train_accuracy
    # print "Testing Accuracy: %f" % test_accuracy
    #
    # plt.figure()
    # # plt.subplot(1, 3, 1)
    # plt.plot(round_errors)
    # # plt.subplot(1, 3, 2)
    # # plt.plot(weighted_train_error)
    # # plt.subplot(1, 3, 3)
    # # plt.plot(weighted_test_error)
    # plt.show()

    trees = []
    testing_data = split_spambase[0]
    training_data = np.array([item for sublist in [x for ind, x in enumerate(split_spambase) if ind != 0] for item in sublist])

    for i in range(50):
        N = 400
        # N = np.shape(training_data)[0]
        b = np.zeros(shape=(N, 58))

        for j in range(N):
            b[j] = random.choice(training_data)

        h = optimal_decision_stump(b[:,:-1], b[:,-1], np.ones(N)/N)
        trees.append(h)

    test_acc = 0.0
    for record in testing_data:
        y_obs = record[-1]
        x_obs = np.matrix(record[:-1])

        pred = np.sign(np.mean(map(lambda h: apply_decision_stump(record, h), trees)))

        if pred == y_obs:
            test_acc += 1

    test_accuracy = test_acc/(np.shape(testing_data)[0])

    print "Bagging Testing Accuracy: %f" % test_accuracy


