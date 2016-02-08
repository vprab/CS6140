__author__ = 'Vishant'
from pylab import *
import numpy as np
from MNIST import load_mnist
from sklearn.multiclass import OutputCodeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

import random
# imshow(images.mean(axis=0), cmap=cm.gray)
# show()


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
    images, labels = load_mnist('training', selection=slice(0, 59999, 10), path='.')
    test_images, test_labels = load_mnist('testing', path='.')

    rectangles = []
    for c in range(100):
        top_left = random.choice([(i, j) for i in range(23) for j in range(23)])
        bottom_right = random.choice([(i, j) for i in range(5, 28) for j in range(5, 28) if i > top_left[0]+4 if j > top_left[1]+4])

        rectangles.append((top_left, bottom_right))

    train_ecoc_table = np.zeros(shape=(np.shape(images)[0], 200))
    for ind, im in enumerate(images):
        row = []
        for (top_left, bottom_right) in rectangles:
            row += get_haar_features(im, top_left, bottom_right)

        train_ecoc_table[ind] = row

    test_ecoc_table = np.zeros(shape=(np.shape(test_images)[0], 200))
    for ind, im in enumerate(test_images):
        row = []
        for (top_left, bottom_right) in rectangles:
            row += get_haar_features(im, top_left, bottom_right)

        test_ecoc_table[ind] = row

    clf = OutputCodeClassifier(AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200), code_size=5, random_state=0)
    clf.fit(train_ecoc_table, labels)

    train_pred = np.array(clf.predict(train_ecoc_table))
    print "Digits Training Accuracy: %f" % (np.sum(train_pred == np.array(labels)).astype(np.float)/np.shape(train_pred)[0])

    test_pred = np.array(clf.predict(test_ecoc_table))
    print "Digits Testing Accuracy: %f" % (np.sum(test_pred == np.array(test_labels)).astype(np.float)/np.shape(test_pred)[0])

    # ecoc_table = []
    # for im in images:
    #
    #     im_preprocess = np.matrix([[np.sum(im[:i,:j]) for i in range(1, 29)] for j in range(1, 29)])
    #
    #     def get_black_rectangle(top_left, bottom_right):
    #         x1, y1 = top_left
    #         x2, y2 = bottom_right
    #
    #         return im_preprocess[x2, y2] - im_preprocess[x2, y1] - im_preprocess[x1, y2] + im_preprocess[x1, y1]
    #
    #     def get_haar_features(top_left, bottom_right):
    #         x1, y1 = top_left
    #         x2, y2 = bottom_right
    #         x_mid, y_mid = (x1+x2)/2, (y1+y2)/2
    #
    #         haar_horizontal = get_black_rectangle(top_left, (x2, y_mid)) - get_black_rectangle((x1, y_mid), bottom_right)
    #         haar_vertical = get_black_rectangle(top_left, (x_mid, y2)) - get_black_rectangle((x_mid, y1), bottom_right)
    #
    #         return [haar_horizontal, haar_vertical]
    #
    #     im_haar_features = []
    #     for r in rectangles:
    #         im_haar_features += get_haar_features(r[0], r[1])
    #
    #     ecoc_table.append(im_haar_features)
    #
    # ecoc_table = np.array(ecoc_table)
    #
    # test_ecoc_table = []
    # for im in test_images:
    #
    #     im_preprocess = np.matrix([[np.sum(im[:i,:j]) for i in range(1, 29)] for j in range(1, 29)])
    #
    #     def get_black_rectangle(top_left, bottom_right):
    #         x1, y1 = top_left
    #         x2, y2 = bottom_right
    #
    #         return im_preprocess[x2, y2] - im_preprocess[x2, y1] - im_preprocess[x1, y2] + im_preprocess[x1, y1]
    #
    #     def get_haar_features(top_left, bottom_right):
    #         x1, y1 = top_left
    #         x2, y2 = bottom_right
    #         x_mid, y_mid = (x1+x2)/2, (y1+y2)/2
    #
    #         haar_horizontal = get_black_rectangle(top_left, (x2, y_mid)) - get_black_rectangle((x1, y_mid), bottom_right)
    #         haar_vertical = get_black_rectangle(top_left, (x_mid, y2)) - get_black_rectangle((x_mid, y1), bottom_right)
    #
    #         return [haar_horizontal, haar_vertical]
    #
    #     im_haar_features = []
    #     for r in rectangles:
    #         im_haar_features += get_haar_features(r[0], r[1])
    #
    #     test_ecoc_table.append(im_haar_features)
    #
    # test_ecoc_table = np.array(test_ecoc_table)
    #
    # clf = OutputCodeClassifier(AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200), code_size=5, random_state=0)
    # clf.fit(ecoc_table, labels)
    #
    # train_pred = np.matrix(clf.predict(ecoc_table)).T
    # print np.shape(train_pred)
    # print np.shape(labels)
    # print "ECOC Training Accuracy: %f" % (np.sum(train_pred == labels).astype(np.float)/np.shape(labels)[0])
    #
    # test_pred = np.matrix(clf.predict(test_ecoc_table)).T
    # print "ECOC Testing Accuracy: %f" % (np.sum(test_pred == labels).astype(np.float)/np.shape(test_labels)[0])



