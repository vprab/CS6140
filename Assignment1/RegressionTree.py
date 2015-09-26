__author__ = 'Sree Vishant Prabhakaran'
import math
import numpy as np


# -------------------------------------------------------------------------------------------------------------------- #
#                                                   PROBLEM 1                                                          #
# -------------------------------------------------------------------------------------------------------------------- #


def get_values(data, attr):
    """

    :param data: The dataset
    :param attr: The column number of a particular attribute within this dataset
    :return: All of the unique values for the given attribute in the dataset
    """
    return set([record[attr] for record in data])


def get_split(data, attr, val):
    """

    :param data: The dataset
    :param attr: The column number of a particular attribute within this dataset
    :param val: A particular value for attr in this dataset
    :return: [G,L], where G is a list of the records in the dataset where the value at the given attribute
    is greater than val, and L is a list of the records where the value of the given attribute is less than val
    """
    return [[record for record in data if record[attr] >= val], [record for record in data if record[attr] < val]]


def entropy(data, target_attr):
    """

    :param data: The dataset
    :param target_attr: The column number representing the target attribute within the dataset
    :return: The entropy of the dataset with respect to the target attribute
    """
    target_col = [record[target_attr] for record in data]
    val_freq = {val:float(target_col.count(val)) for val in set(target_col)}
    entropy_val = sum(map(lambda f: (-f/len(data)) * math.log(f/len(data), 2), val_freq.values()))

    return entropy_val


def gain(data, attr, val, target_attr):
    """

    :param data: The dataset
    :param attr: The column number representing a particular attribute within the dataset
    :param val: A particular value for the given attribute
    :param target_attr: The column number representing the target attribute within the dataset
    :return: The information gain if we were to split the dataset on attr at val
    """

    above, below = get_split(data, attr, val)
    above_entropy, above_size = entropy(above, target_attr), len(above)
    below_entropy, below_size = entropy(below, target_attr), len(below)
    total = len(data)

    child_entropy = (above_entropy*above_size/total) + (below_entropy*below_size/total)
    return entropy(data, target_attr) - child_entropy


def mean_squared_error_regression(data, target_attr):
    target_col = [record[target_attr] for record in data]

    if not data:
        return 0
    else:
        mu = np.mean(target_col)
        mse = np.mean([(val - mu)**2 for val in target_col])

        return mse


def gain_regression(data, attr, val, target_attr):

    if data:
        data_entropy = mean_squared_error_regression(data, target_attr)
        split = get_split(data, attr, val)
        child_entropy = np.sum([mean_squared_error_regression(d, target_attr)*len(d) for d in split])
        child_entropy /= len(data)

        return data_entropy - child_entropy
    else:
        return 0


def choose_attribute_and_value(data, attributes, target_attr, regression):
    """

    :param data: The dataset
    :param attributes: The list of column numbers representing all attributes in this dataset (except for the target)
    :param target_attr: The column number representing the target attribute
    :return: [A, V] where A is the column number of the best attribute to branch on based on information gain,
    and V is the particular value of this attribute to branch on
    """

    ans_attr = attributes[0]
    ans_val = data[attributes[0]][0]
    max_gain = 0

    for attr in attributes:
        vals = get_values(data, attr)

        for val in vals:
            if regression:
                g = gain_regression(data, attr, val, target_attr)
            else:
                g = gain(data, attr, val, target_attr)

            if g > max_gain:
                max_gain = g
                ans_attr = attr
                ans_val = val

    return [ans_attr, ans_val]


def create_decision_tree(data, attributes, target_attr, regression, max_levels=4, alpha=0.5, current_level=0):
    """

    :param data: The dataset
    :param attributes: The list of column numbers representing all attribuets in this dataset (except for the target)
    :param target_attr: The column number representing the target attribute
    :param regression: True for regression tree, False for decision tree
    :param max_levels: The maximum number of levels in this tree (not including the root)
    :param alpha: Do not split a node if its entropy is alpha or less
    :param current_level: The current level of recursion that this function is at (do not use)
    :return: A decision tree object
    """

    if not data:
        return 0

    data = data[:]
    vals = [record[target_attr] for record in data]
    default = max(set(vals), key=vals.count)

    # if not data or len(attributes) == 0:
    #     return default
    if current_level == max_levels:
        if regression:
            return np.mean(vals)
        else:
            return default
    else:
        best_attr, best_val = choose_attribute_and_value(data, attributes, target_attr, regression)
        tree = {(best_attr, best_val): {}}

        tree[(best_attr, best_val)][1], tree[(best_attr, best_val)][0] = map(lambda d: create_decision_tree(d, attributes, target_attr, regression, max_levels, alpha=alpha, current_level=current_level+1), get_split(data, best_attr, best_val))

    return tree


def print_decision_tree(t, atts, level=0):
    """

    :param t: The tree to print
    :param atts: The attribute labels for the dataset that the tree was trained on
    :param level: The current level of recursion that this function is at (do not use)
    :return:
    """

    if isinstance(t[t.keys()[0]][1], (int, long, float)):
        print '| '*level + atts[t.keys()[0][0]] + ' >= ' + str(t.keys()[0][1]) + ' : ' + str(t[t.keys()[0]][1])
    else:
        print '| '*level + atts[t.keys()[0][0]] + ' >= ' + str(t.keys()[0][1])
        print_decision_tree(t[t.keys()[0]][1], level+1)

    if isinstance(t[t.keys()[0]][0], (int, long, float)):
        print '| '*level + atts[t.keys()[0][0]] + ' < ' + str(t.keys()[0][1]) + ' : ' + str(t[t.keys()[0]][0])
    else:
        print '| '*level + atts[t.keys()[0][0]] + ' < ' + str(t.keys()[0][1])
        print_decision_tree(t[t.keys()[0]][0], level+1)


def apply_decision_tree(tree, record):
    """

    :param tree: A decision tree object
    :param record: A record of data
    :return: The result of running the given tree on the record
    """
    if record[tree.keys()[0][0]] >= tree.keys()[0][1]:
        if isinstance(tree[tree.keys()[0]][1], (int, long, float)):
            return tree[tree.keys()[0]][1]
        else:
            return apply_decision_tree(tree[tree.keys()[0]][1], record)
    else:
        if isinstance(tree[tree.keys()[0]][0], (int, long, float)):
            return tree[tree.keys()[0]][0]
        else:
            return apply_decision_tree(tree[tree.keys()[0]][0], record)


def mean_squared_error(data, tree, target_attr):
    """

    :param data: A dataset
    :param tree: A decision tree
    :param target_attr: The column number representing the target attribute
    :return: The MSE between the results of running the given decision tree on the dataset and the actual target values
    """
    return np.mean(map(lambda rec: (apply_decision_tree(tree, rec) - rec[target_attr])**2, data))


with open("housing_train.txt") as housing_train:
    housing_train_data = [line.split() for line in housing_train]

housing_train_data = [[float(x) for x in y] for y in housing_train_data]

with open("housing_test.txt") as housing_test:
    housing_test_data = [line.split() for line in housing_test]

housing_test_data = [[float(x) for x in y] for y in housing_test_data]

housing_decision_tree = create_decision_tree(housing_train_data, range(13), 13, True, max_levels=2)
# housing_atts = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# print_decision_tree(housing_decision_tree, housing_atts)
print "Housing Training Error: %f" % mean_squared_error(housing_train_data, housing_decision_tree, 13)
print "Housing Testing Error: %f" % mean_squared_error(housing_test_data, housing_decision_tree, 13)

# ----------------------------------------------------------------------------------------------------------------------

with open("spambase.data") as spambase:
    spambase_data = [line.split(',') for line in spambase]

spambase_data = [[float(x) for x in y] for y in spambase_data]

K = 10
split_spambase = [spambase_data[i::K] for i in range(K)]

# training_errors = []
# testing_errors = []
#
# for i in range(K):
#     testing_data = split_spambase[i]
#     training_data = [item for sublist in [x for ind, x in enumerate(split_spambase) if ind != i] for item in sublist]
#
#     spam_decision_tree = create_decision_tree(training_data, range(57), 57, False, max_levels=5)
#
#     train_error = mean_squared_error(training_data, spam_decision_tree, 57)
#     test_error = mean_squared_error(testing_data, spam_decision_tree, 57)
#
#     print "Spam Training Error %i : %f" %(i, train_error)
#     print "Spam Testing Error %i : %f" %(i, test_error)
#
#     training_errors.append(train_error)
#     testing_errors.append(test_error)
#
# print "Spam Average Training Error: %f" % np.mean(training_errors)
# print "Spam Average Testing Error: %f" % np.mean(testing_errors)

# -------------------------------------------------------------------------------------------------------------------- #
#                                                   PROBLEM 2                                                          #
# -------------------------------------------------------------------------------------------------------------------- #


def linear_regression(data, target_attr):
    """

    :param data: A dataset
    :param target_attr: The column number representing the target attribute
    :return: A linear regression vector of coefficients
    """
    y = np.matrix([record[target_attr] for record in data])
    x = np.matrix([[val for ind, val in enumerate(record) if ind != target_attr] for record in data])

    b = (x.T * x).I * x.T * y.T

    return b


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

# housing_lin_reg = linear_regression(housing_train_data, 13)
# print "Housing Training Error LR: %f" % mean_squared_error_lr(housing_train_data, housing_lin_reg, 13)
# print "Housing Testing Error LR: %f" % mean_squared_error_lr(housing_test_data, housing_lin_reg, 13)
#
# training_errors_lr = []
# testing_errors_lr = []
#
# for i in range(K):
#     testing_data = split_spambase[i]
#     training_data = [item for sublist in [x for ind, x in enumerate(split_spambase) if ind != i] for item in sublist]
#
#     spam_lin_reg = linear_regression(training_data, 57)
#     training_errors_lr.append(mean_squared_error_lr(training_data, spam_lin_reg, 57))
#     testing_errors_lr.append(mean_squared_error_lr(testing_data, spam_lin_reg, 57))
#
# print "Spam Average Training Error LR: %f" % np.mean(training_errors_lr)
# print "Spam Average Testing Error LR: %f" % np.mean(testing_errors_lr)

# -------------------------------------------------------------------------------------------------------------------- #
#                                                   PROBLEM 3                                                          #
# -------------------------------------------------------------------------------------------------------------------- #

'''
Ch. 8, Problem 1: Suppose we have a decision tree with binary splits on feature f with threshold t at nodes n_p and n_c,
where n_c is a descendant of n_p. Then at n_c, we know that all of the training records will fall on the same side of
the split that they did at n_p. This implies that the split at n_c is hardly a split at all - it does not separate the
data in any way. Therefore, we could simply remove the node at n_c and connect the child of n_c to the parent of n_c,
yielding an equivalent tree with distinct splits on each path.

Ch. 8, Problem 2:

a) We start by noting that the children of the root node R of a tree are themselves the roots of subtrees. Therefore,
we can use induction and simply show that if we can replace R with binary nodes, then any arbitrary tree can be
represented as a binary tree.

Base case: If R has 2 or fewer children, the tree is already binary and we are done. If R has 3 children, we can convert
it as follows:

                R                           R
               /|\      -------->          /  \
              / | \                       C1  not(C1)
            C1  C2 C3                           / \
                                               C2 C3

And if R has more than 3 children, we can simply use the same idea and repeat - R with children C1 and not(C1), not(C1)
with children C2 and not(C2), and so on.

Now by induction, if any of the children of R are not binary trees, we can simply use this process until the whole tree
is binary.

b) Lower bound of 2, upper bound of ceiling(log_2 B)

c) Lower bound of 3, upper bound of 2B-1
'''

# -------------------------------------------------------------------------------------------------------------------- #
#                                                   PROBLEM 4                                                          #
# -------------------------------------------------------------------------------------------------------------------- #

'''
Ch. 8, Problem 5:

a)
'''
