__author__ = 'Vishant'
import numpy as np
import copy
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from DecisionStumpAdaboost import *

codes = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0],
[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0],
[0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0],
[0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
[1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0],
[1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
[0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1]])

train_data = np.zeros(shape=(11314, 1755))
with open("8newsgroup/train.trec/feature_matrix.txt") as train_matrix:
        for ind, line in enumerate(train_matrix):
            arr = line.split()

            train_data[ind, -1] = int(arr[0])

            for s in arr[1:]:
                f, val = s.split(':')
                f = int(f)
                val = float(val)

                train_data[ind, f] = val

classifiers = []

for i in range(np.shape(codes)[1]):
    map_classes = codes[:, i]
    data_copy = copy.deepcopy(train_data)

    for ind, val in enumerate(train_data[:,-1]):
        if map_classes[val]:
            data_copy[ind, -1] = 1
        else:
            data_copy[ind, -1] = -1

    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators = 100)

    # h = adaboost_decision_stump(data_copy, 1754, random_decision_stump, 200)

    bdt.fit(data_copy[:,:-1], data_copy[:,-1])

    classifiers.append(bdt)

test_data = np.zeros(shape=(7532, 1755))
with open("8newsgroup/test.trec/feature_matrix.txt") as test_matrix:
        for ind, line in enumerate(test_matrix):
            arr = line.split()

            test_data[ind, -1] = int(arr[0])

            for s in arr[1:]:
                f, val = s.split(':')
                f = int(f)
                val = float(val)

                test_data[ind, f] = val

def ecoc_sign(num):
    if np.sign(num) == -1:
        return 0
    else:
        return np.sign(num)

test_acc = 0.0
for record in test_data:
    # rec_codes = map(lambda h: ecoc_sign(apply_adaboost_decision_stump(record[:-1], h)), classifiers)
    rec_codes = map(lambda bdt: ecoc_sign(bdt.predict(record[:-1])[0]), classifiers)
    rec_codes = np.array(rec_codes)
    pred_class = np.argmax(np.sum(rec_codes == codes, axis=1))

    # print record[-1]

    if pred_class == record[-1]:
        test_acc += 1.0/7532

print "Testing Accuracy: %f" % test_acc


