__author__ = 'Vishant'
import numpy as np
import random
from DecisionStumpAdaboost import *


def read_data(file):
    map_dict = {}
    discrete = []
    with open(file + "/" + file + ".config") as config:
        for ind, line in enumerate(config):
            arr = line.split()

            if ind == 0:
                m = int(arr[0])
                n = int(arr[1]) + int(arr[2]) + 1
            elif arr != ['1']:
                if arr[0] != '-1000':
                    discrete.append(ind-1)
                    map_dict[ind-1] = {}
                    for i in range(int(arr[0])):
                        map_dict[ind-1][arr[1+i]] = i

    output = np.zeros(shape=(m, n))
    with open(file + "/" + file + ".data") as data:
        for row, line in enumerate(data):
            arr = line.split()

            for col, val in enumerate(arr):
                if val == '?':
                    if col in map_dict:
                        output[row, col] = map_dict[col][random.choice(map_dict[col].keys())]
                    else:
                        output[row, col] = np.mean(output[:row, col])
                elif col in map_dict:
                    output[row, col] = map_dict[col][val]
                else:
                    output[row, col] = val

    return output, map_dict, discrete

# crx_data, crx_dict, crx_discrete = read_data('crx')
vote_data, vote_dict, vote_discrete = read_data('vote')

def optimal_decision_stump_uci(x, y, distribution, discrete):
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

                if val not in discrete:
                    if comp_val < val:
                        error += dict_attr_val[1]
                    else:
                        error += dict_attr_val[0]
                else:
                    if comp_val != val:
                        error += dict_attr_val[1]
                    else:
                        error += dict_attr_val[0]

            true_error = np.abs(1/2.0 - error)
            if true_error > opt_error:
                opt_error = true_error
                ans_attr = attr
                ans_val = val

    return ans_attr, ans_val

def apply_decision_stump_uci(record, stump, discrete):
    attr, val = stump

    if val in discrete:
        if record[attr] == val:
            return 1
        else:
            return -1
    else:
        if record[attr] >= val:
            return 1
        else:
            return -1

def apply_adaboost_decision_stump_uci(record, ada_stump, discrete):
    ans = 0

    for alpha, stump in ada_stump:
        ans += alpha*apply_decision_stump_uci(record, stump, discrete)

    return ans

if __name__ == "__main__":

    K = 10
    # split_crx = np.array([crx_data[i::K] for i in range(K)])
    split_vote = np.array([vote_data[i::K] for i in range(K)])

    round_errors = []

    testing_data = split_vote[8]
    training_data = np.array([item for sublist in [x for ind, x in enumerate(split_vote) if ind != 8] for item in sublist])

    m, n = np.shape(training_data)
    y = training_data[:,16]
    x = training_data[:, [i for i in range(n) if i != 16]]
    distribution = np.array([1.0/m]*m)

    ans = []
    for t in range(0, 100):
        h = optimal_decision_stump_uci(x, y, distribution, vote_discrete)
        # h = random_decision_stump(x, y, distribution)

        epsilon = 0.0
        local_round_error = 0.0
        for ind, record in enumerate(x):
            pred_val = apply_decision_stump_uci(record, h, vote_discrete)
            if (pred_val == -1 and y[ind] == 1) or (pred_val == 1 and y[ind] == 0):
                epsilon += distribution[ind]
                local_round_error += distribution[ind]

        round_errors.append(local_round_error)
        alpha = 1/2.0 * np.log(max((1-epsilon)/epsilon, 1e-300))

        for ind, record in enumerate(x):
            distribution[ind] *= np.exp(-alpha*y[ind]*apply_decision_stump(record, h))

        distribution /= np.sum(distribution, axis=0)
        ans.append((alpha, h))

    train_acc = 0.0
    for record in training_data:
        y_obs = record[16]
        x_obs = np.matrix(record[:16])
        pred_val = np.sign(apply_adaboost_decision_stump_uci(record, ans, vote_discrete))

        if (pred_val == -1 and y_obs == 0) or (pred_val == 1 and y_obs == 1):
            train_acc += 1

    test_acc = 0.0
    for record in testing_data:
        y_obs = record[16]
        x_obs = np.matrix(record[:16])
        pred_val = np.sign(apply_adaboost_decision_stump_uci(record, ans, vote_discrete))

        if (pred_val == -1 and y_obs == 0) or (pred_val == 1 and y_obs == 1):
            test_acc += 1

    train_accuracy = train_acc/(np.shape(training_data)[0])
    test_accuracy = test_acc/(np.shape(testing_data)[0])


    print "Training Accuracy: %f" % train_accuracy
    print "Testing Accuracy: %f" % test_accuracy

    plt.figure()
    plt.plot(round_errors)
    plt.show()
