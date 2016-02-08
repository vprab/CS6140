__author__ = 'Vishant'
import numpy as np
import random
from sklearn import svm
from MNIST import *


#### LOADING DATA ####


with open("spambase.data") as spambase:
    spambase_data = [line.split(',') for line in spambase]

spambase_data = np.array(spambase_data).astype(np.float)
for i in range(np.shape(spambase_data)[0]):
    if spambase_data[i][-1] == 0:
        spambase_data[i][-1] = -1

K = 10
split_spambase = np.array([spambase_data[i::K] for i in range(K)])

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


#### SMO SOLVER ####


# def SMO(c, tol, iterations, train, labels):
#     m, n = np.shape(train)
#     alpha = np.zeros(shape=(m))
#     b = 0.0
#
#     num_passes = 0
#     while num_passes < iterations:
#         num_changed_alphas = 0
#         for i in range(m):
#             E_i = b
#             for k in range(m):
#                 E_i += alpha[k]*labels[k]*np.dot(train[i], train[k])
#
#             E_i -= labels[i]
#
#             if (labels[i]*E_i < -tol and alpha[i] < c) or (labels[i]*E_i > tol and alpha[i] > 0):
#                 j = random.choice([k for k in range(m) if k != i])
#                 E_j = b
#                 for k in range(m):
#                     E_j += alpha[k]*labels[k]*np.dot(train[j], train[k])
#
#                 E_j -= labels[j]
#
#                 alpha_i_old = alpha[i]
#                 alpha_j_old = alpha[j]
#
#                 if labels[i] != labels[j]:
#                     L = max(0, alpha[j] - alpha[i])
#                     H = min(c, c + alpha[j] - alpha[i])
#                 else:
#                     L = max(0, alpha[i] + alpha[j] - c)
#                     H = min(c, alpha[i] + alpha[j])
#
#                 if L == H:
#                     break
#
#                 eta = 2*np.dot(train[i], train[j]) - np.dot(train[i], train[i]) - np.dot(train[j], train[j])
#
#                 if eta > 0:
#                     break
#
#                 alpha[j] -= labels[j]*(E_i - E_j)/eta
#
#                 if alpha[j] > H:
#                     alpha[j] = H
#                 if alpha[j] < L:
#                     alpha[j] = L
#
#                 if abs(alpha[j] - alpha_j_old) < 1e-5:
#                     break
#
#                 alpha[i] += labels[i]*labels[j]*(alpha_j_old - alpha[j])
#
#                 b1 = b - E_i - labels[i]*(alpha[i] - alpha_i_old)*np.dot(train[i], train[i]) - labels[j]*(alpha[j] - alpha_j_old)*np.dot(train[i], train[j])
#                 b2 = b - E_j - labels[i]*(alpha[i] - alpha_i_old)*np.dot(train[i], train[j]) - labels[j]*(alpha[j] - alpha_j_old)*np.dot(train[j], train[j])
#
#                 if 0 < alpha[i] < c:
#                     b = b1
#                 if 0 < alpha[j] < c:
#                     b = b2
#                 else:
#                     b = (b1+b2)/2.0
#
#                 num_changed_alphas += 1
#
#         if num_changed_alphas == 0:
#             num_passes += 1
#         else:
#             num_passes = 0
#
#     return train, labels, alpha, b


def apply_smo(smo, record):
    train, labels, alpha, b = smo
    m, n = np.shape(train)
    val = b

    for i in range(m):
        val += alpha[i]*labels[i]*np.dot(train[i], record)

    if val >= 0:
        return 1
    else:
        return -1


def smoSimple(dataIn, classLabels, C, tolerance, maxIter):
    dataMatrix = np.matrix(dataIn)
    labelMat = np.matrix(classLabels).T
    m, n = np.shape(dataMatrix)
    alphas = np.zeros((m))

    bias = 0

    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):

            # Evaluate the model i
            fXi = bias
            for k in range(m):
                fXi += alphas[k]*labelMat[k]*np.dot(dataMatrix[i], dataMatrix[k].T)
            # fXi = evaluate(i, alphas, labelMat, dataMatrix, bias)
            Ei = fXi - float(labelMat[i])

            # Check if we can optimize (alphas always between 0 and C)
            if ((labelMat[i] * Ei < -tolerance) and (alphas[i] < C)) or \
                ((labelMat[i] * Ei > tolerance) and (alphas[i] > 0)):

                # Select a random J
                j = random.choice([k for k in range(m) if k != i])

                # Evaluate the mode j
                fXj = bias
                for k in range(m):
                    fXj += alphas[k]*labelMat[k]*np.dot(dataMatrix[j], dataMatrix[k].T)
                # fXj = evaluate(j, alphas, labelMat, dataMatrix, bias)
                Ej = fXj - float(labelMat[j])

                # Copy alphas
                alpha_old_i = alphas[i].copy()
                alpha_old_j = alphas[j].copy()

                # Check how much we can change the alphas
                # L = Lower bound
                # H = Higher bound
                if labelMat[i] != labelMat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                # If the two correspond, then there is nothing
                # we can really do
                if L == H:
                    print "L is H"
                    continue

                # Calculate ETA
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                        dataMatrix[i, :] * dataMatrix[i, :].T - \
                        dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print "eta is bigger than 0"
                    continue

                # Update J and I alphas
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta

                if alphas[j] > H:
                    alphas[j] = H
                if alphas[j] < L:
                    alphas[j] = L

                # alphas[j] = clipAlpha(alphas[j], H, L)

                # If alpha is not moving enough, continue..
                if abs(alphas[j] - alpha_old_j) < 0.001:
                    print "Alpha not moving too much.."
                    continue
                # Change alpha I for the exact value, in the opposite
                # direction
                alphas[i] += labelMat[j] * labelMat[i] * \
                        (alpha_old_j - alphas[j])

                # Update bias
                b1 = bias - Ei - labelMat[i] * (alphas[i] - alpha_old_i) * \
                        dataMatrix[i, :] * dataMatrix[i, :].T - \
                        labelMat[j] * (alphas[j]-alpha_old_j) * \
                        dataMatrix[i, :] * dataMatrix[j, :].T

                b2 = bias - Ej - labelMat[i] * (alphas[i] - alpha_old_i) * \
                        dataMatrix[i, :] * dataMatrix[i, :].T - \
                        labelMat[j] * (alphas[j]-alpha_old_j) * \
                        dataMatrix[j, :] * dataMatrix[j, :].T

                # Choose bias to set
                if 0 < alphas[i] and C > alphas[i]:
                    bias = b1
                elif 0 < alphas[j] and C > alphas[j]:
                    bias = b2
                else:
                    bias = (b1 + b2) / 2.0

                # Increment counter and log
                alphaPairsChanged += 1
                print "iter: %d i:%d, pairs changed %d" % (
                    iter, i, alphaPairsChanged
                )

            if alphaPairsChanged == 0:
                iter += 1
            else:
                iter = 0
            print "Iteration number: %s" % iter

        print alphas[alphas>0]
        print bias

    return dataIn, classLabels, alphas, bias

if __name__ == "__main__":
    # testing_data = split_spambase[0]
    # training_data = np.array([item for sublist in [x for ind, x in enumerate(split_spambase) if ind != 0] for item in sublist])
    #
    # # spambase_smo = SMO(0.01, 0.01, 100, training_data[:,:-1], training_data[:,-1])
    # spambase_smo = smoSimple(training_data[:,:-1], training_data[:,-1], 0.01, 0.01, 100)
    #
    # m, n = np.shape(testing_data)
    # test_acc = 0.0
    # for record in testing_data:
    #     if apply_smo(spambase_smo, record[:-1]) == record[-1]:
    #         test_acc += 1.0/m
    #
    # print "Simplified SMO Test Accuracy: ", test_acc

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

    digits_smo = smoSimple(train_ecoc_table[:,:-1], train_ecoc_table[:,-1], 0.01, 0.01, 100)
    test_acc = 0
    m, n = np.shape(test_ecoc_table)
    for ind, record in enumerate(test_ecoc_table):
        if apply_smo(digits_smo, record) == test_labels[ind]:
            test_acc += 1.0/m

    print "Simplified SMO Digits Accuracy: ", test_acc