__author__ = 'Vishant'
from RegressionTree import create_decision_tree, apply_decision_tree, mean_squared_error
import numpy as np
import copy

with open("housing_train.txt") as housing_train:
    housing_train_data = [line.split() for line in housing_train]

housing_train_data = [[float(x) for x in y] for y in housing_train_data]

with open("housing_test.txt") as housing_test:
    housing_test_data = [line.split() for line in housing_test]

housing_test_data = [[float(x) for x in y] for y in housing_test_data]


def apply_adaboost_regression_trees(ada_trees, record):
    return sum(map(lambda f: apply_decision_tree(f, record), ada_trees))


def adaboost_mean_squared_error(data, ada_trees, target_attr):
    return np.mean(map(lambda rec: (apply_adaboost_regression_trees(ada_trees, rec) - rec[target_attr])**2, data))

ans = []
housing_train_data_copy = copy.deepcopy(housing_train_data)
for i in range(1, 11):
    reg_tree = create_decision_tree(housing_train_data_copy, range(13), 13, True, max_levels=2)

    for ind, record in enumerate(housing_train_data_copy):
        y_pred = apply_decision_tree(reg_tree, record)
        housing_train_data_copy[ind][13] -= y_pred

    print "Tree Training Error: %f" % mean_squared_error(housing_train_data_copy, reg_tree, 13)

    ans.append(reg_tree)

print "Training Error: %f" % adaboost_mean_squared_error(housing_train_data, ans, 13)
print "Testing Error: %f" % adaboost_mean_squared_error(housing_test_data, ans, 13)
