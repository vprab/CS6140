__author__ = 'Vishant'
from sklearn import svm
import numpy as np
from MNIST import *
import random

with open("spambase.data") as spambase:
    spambase_data = [line.split(',') for line in spambase]

spambase_data = np.array(spambase_data).astype(np.float)
for i in range(np.shape(spambase_data)[0]):
    if spambase_data[i][-1] == 0:
        spambase_data[i][-1] = -1

K = 10
split_spambase = np.array([spambase_data[i::K] for i in range(K)])

testing_spambase = split_spambase[0]
training_spambase = np.array([item for sublist in [x for ind, x in enumerate(split_spambase) if ind != 0] for item in sublist])

# clf = svm.SVC(kernel='poly')
# clf.fit(training_spambase[:,:-1], training_spambase[:, -1])
# predictions = clf.predict(testing_spambase[:, :-1])
#
# print "SVM Polynomial Kernel Spambase Testing Accuracy: %f" % (np.sum(predictions == testing_spambase[:,-1])/float(len(predictions)))

# clf = svm.SVC(kernel='linear')
# clf.fit(training_spambase[:,:-1], training_spambase[:, -1])
# predictions = clf.predict(testing_spambase[:, :-1])
#
# print "SVM Linear Kernel Spambase Testing Accuracy: %f" % (np.sum(predictions == testing_spambase[:,-1])/float(len(predictions)))

##############################################

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
    clf = svm.SVC()
    clf.fit(training_spambase[:,:-1], training_spambase[:, -1])
    predictions = clf.predict(testing_spambase[:, :-1])

    print "SVM RBF Kernel Spambase Testing Accuracy: %f" % (np.sum(predictions == testing_spambase[:,-1])/float(len(predictions)))

    clf = svm.LinearSVC()
    clf.fit(training_spambase[:,:-1], training_spambase[:, -1])
    predictions = clf.predict(testing_spambase[:, :-1])

    print "SVM Linear Kernel Spambase Testing Accuracy: %f" % (np.sum(predictions == testing_spambase[:,-1])/float(len(predictions)))

    clf = svm.SVC(kernel='poly')
    clf.fit(training_spambase[:,:-1], training_spambase[:, -1])
    predictions = clf.predict(testing_spambase[:, :-1])

    print "SVM Polynomial Kernel Spambase Testing Accuracy: %f" % (np.sum(predictions == testing_spambase[:,-1])/float(len(predictions)))


    # images, labels = load_mnist('training', selection=slice(0, 59999, 10), path='.')
    # test_images, test_labels = load_mnist('testing', path='.')
    #
    # rectangles = []
    # for c in range(100):
    #     top_left = random.choice([(i, j) for i in range(23) for j in range(23)])
    #     bottom_right = random.choice([(i, j) for i in range(5, 28) for j in range(5, 28) if i > top_left[0]+4 if j > top_left[1]+4])
    #
    #     rectangles.append((top_left, bottom_right))
    #
    # train_ecoc_table = np.zeros(shape=(np.shape(images)[0], 200))
    # for ind, im in enumerate(images):
    #     row = []
    #     for (top_left, bottom_right) in rectangles:
    #         row += get_haar_features(im, top_left, bottom_right)
    #
    #     train_ecoc_table[ind] = row
    #
    # test_ecoc_table = np.zeros(shape=(np.shape(test_images)[0], 200))
    # for ind, im in enumerate(test_images):
    #     row = []
    #     for (top_left, bottom_right) in rectangles:
    #         row += get_haar_features(im, top_left, bottom_right)
    #
    #     test_ecoc_table[ind] = row
    #
    # clf = svm.SVC()
    # clf.fit(train_ecoc_table, labels)
    # predictions = np.array(clf.predict(train_ecoc_table))
    #
    # print "SVM Digits Training Accuracy: %f" % (np.sum(predictions == np.array(labels)).astype(np.float)/np.shape(predictions)[0])