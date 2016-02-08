__author__ = 'Vishant'
import numpy as np
from MNIST import load_mnist
import random


def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=2):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=1.0):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))


#### LOADING DATA ####

with open("spambase.data") as spambase:
    spambase_data = [line.split(',') for line in spambase]

spambase_data = np.array(spambase_data).astype(np.float)
for i in range(np.shape(spambase_data)[0]):
    if spambase_data[i][-1] == 0:
        spambase_data[i][-1] = -1

K = 10
split_spambase = np.array([spambase_data[i::K] for i in range(K)])
spambase_data_norm = np.copy(spambase_data)

def normalize():

    f = len(spambase_data_norm[0]) - 1
    mins = np.array([min(spambase_data_norm[:,i]) for i in range(f)])
    maxs = np.array([max(spambase_data_norm[:,i]) for i in range(f)])

    spambase_data_norm[:, :f] = (spambase_data_norm[:, :f] - mins)/(maxs - mins)

normalize()

split_spambase_norm = np.array([spambase_data_norm[i::K] for i in range(K)])


#### K-NEAREST NEIGHBORS ####


def kNN(train, rec, k, sim_func=None):
    if sim_func is None:
        sim_func = lambda a, b: -np.linalg.norm(a - b)

    neighbors = sorted(train, key=lambda r: -sim_func(rec, r[:-1]))[0:k]
    return max(set(np.array(neighbors)[:,-1]), key=list(np.array(neighbors)[:,-1]).count)


def get_black_rectangle(im, top_left, bottom_right):
    left_x, left_y = top_left
    right_x, right_y = bottom_right

    return np.sum(im[left_x:right_x, left_y:right_y] == 0)


def get_haar_features(im, top_left, bottom_right):
    mid_x = (top_left[0] + bottom_right[0])/2
    mid_y = (top_left[1] + bottom_right[1])/2

    haar_horizontal = get_black_rectangle(im, top_left, (mid_x, bottom_right[1])) - get_black_rectangle(im, (mid_x, top_left[1]), bottom_right)
    haar_vertical = get_black_rectangle(im, top_left, (bottom_right[0], mid_y)) - get_black_rectangle(im, (top_left[0], mid_y), bottom_right)

    return [haar_horizontal, haar_vertical]


if __name__ == "__main__":
    # testing_data = split_spambase_norm[0]
    # training_data = np.array([item for sublist in [x for ind, x in enumerate(split_spambase_norm) if ind != 0] for item in sublist])
    #
    # m, n = np.shape(testing_data)
    # test_acc = 0.0
    # for t in testing_data:
    #     if kNN(training_data, t[:-1], 1) == t[-1]:
    #         test_acc += 1.0/m
    #
    # print "1-Nearest Neighbors Spambase Testing Accuracy: ", test_acc
    #
    # test_acc = 0.0
    # for t in testing_data:
    #     if kNN(training_data, t[:-1], 3) == t[-1]:
    #         test_acc += 1.0/m
    #
    # print "3-Nearest Neighbors Spambase Testing Accuracy: ", test_acc
    #
    # test_acc = 0.0
    # for t in testing_data:
    #     if kNN(training_data, t[:-1], 7) == t[-1]:
    #         test_acc += 1.0/m
    #
    # print "7-Nearest Neighbors Spambase Testing Accuracy: ", test_acc

    images, labels = load_mnist('training', selection=slice(0, 59999, 10), path='.')
    test_images, test_labels = load_mnist('testing', selection=slice(0, 59999, 10), path='.')

    rectangles = []
    for c in range(100):
        top_left = random.choice([(i, j) for i in range(23) for j in range(23)])
        bottom_right = random.choice([(i, j) for i in range(5, 28) for j in range(5, 28) if i > top_left[0]+4 if j > top_left[1]+4])

        rectangles.append((top_left, bottom_right))

    train_ecoc_table = np.zeros(shape=(np.shape(images)[0], 201))
    for ind, im in enumerate(images):
        row = []
        for (top_left, bottom_right) in rectangles:
            row += get_haar_features(im, top_left, bottom_right)

        train_ecoc_table[ind] = row + [labels[ind]]

    test_ecoc_table = np.zeros(shape=(np.shape(test_images)[0], 200))
    for ind, im in enumerate(test_images):
        row = []
        for (top_left, bottom_right) in rectangles:
            row += get_haar_features(im, top_left, bottom_right)

        test_ecoc_table[ind] = row

    m, n = np.shape(test_ecoc_table)
    # test_acc = 0.0
    # for ind, t in enumerate(test_ecoc_table):
    #     if kNN(train_ecoc_table, t, 1, sim_func=lambda x, y: (np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)))) == test_labels[ind]:
    #         test_acc += 1.0/m
    #
    # print "1-Nearest Neighbors Digits Testing Accuracy: ", test_acc
    #
    # test_acc = 0.0
    # for ind, t in enumerate(test_ecoc_table):
    #     if kNN(train_ecoc_table, t, 3, sim_func=lambda x, y: (np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)))) == test_labels[ind]:
    #         test_acc += 1.0/m
    #
    # print "3-Nearest Neighbors Digits Testing Accuracy: ", test_acc
    #
    # test_acc = 0.0
    # for ind, t in enumerate(test_ecoc_table):
    #     if kNN(train_ecoc_table, t, 7, sim_func=lambda x, y: (np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)))) == test_labels[ind]:
    #         test_acc += 1.0/m
    #
    # print "7-Nearest Neighbors Digits Testing Accuracy: ", test_acc

    test_acc = 0.0
    for ind, t in enumerate(test_ecoc_table):
        if kNN(train_ecoc_table, t, 1, sim_func=gaussian_kernel) == test_labels[ind]:
            test_acc += 1.0/m

    print "1-Nearest Neighbors Digits Testing Accuracy: ", test_acc

    test_acc = 0.0
    for ind, t in enumerate(test_ecoc_table):
        if kNN(train_ecoc_table, t, 3, sim_func=gaussian_kernel) == test_labels[ind]:
            test_acc += 1.0/m

    print "3-Nearest Neighbors Digits Testing Accuracy: ", test_acc

    test_acc = 0.0
    for ind, t in enumerate(test_ecoc_table):
        if kNN(train_ecoc_table, t, 7, sim_func=gaussian_kernel) == test_labels[ind]:
            test_acc += 1.0/m

    print "7-Nearest Neighbors Digits Testing Accuracy: ", test_acc

    # test_acc = 0.0
    # for ind, t in enumerate(test_ecoc_table):
    #     if kNN(train_ecoc_table, t, 1, sim_func=lambda x, y: (np.dot(x,y)**2)) == test_labels[ind]:
    #         test_acc += 1.0/m
    #
    # print "1-Nearest Neighbors Digits Testing Accuracy: ", test_acc
    #
    # test_acc = 0.0
    # for ind, t in enumerate(test_ecoc_table):
    #     if kNN(train_ecoc_table, t, 3, sim_func=lambda x, y: (np.dot(x,y)**2)) == test_labels[ind]:
    #         test_acc += 1.0/m
    #
    # print "3-Nearest Neighbors Digits Testing Accuracy: ", test_acc
    #
    # test_acc = 0.0
    # for ind, t in enumerate(test_ecoc_table):
    #     if kNN(train_ecoc_table, t, 7, sim_func=lambda x, y: (np.dot(x,y)**2)) == test_labels[ind]:
    #         test_acc += 1.0/m
    #
    # print "7-Nearest Neighbors Digits Testing Accuracy: ", test_acc