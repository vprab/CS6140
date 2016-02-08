__author__ = 'Vishant'
import numpy as np

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=2):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=1.0):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

with open("perceptronData.txt") as perceptron:
    perceptron_data = [line.split() for line in perceptron]

perceptron_data = np.array(perceptron_data).astype(np.float)
perceptron_data_norm = np.copy(perceptron_data)

def normalize():

    f = len(perceptron_data_norm[0]) - 1
    mins = np.array([min(perceptron_data_norm[:,i]) for i in range(f)])
    maxs = np.array([max(perceptron_data_norm[:,i]) for i in range(f)])

    perceptron_data_norm[:, :f] = (perceptron_data_norm[:, :f] - mins)/(maxs - mins)

normalize()

with open("twoSpirals.txt") as spiral:
    spiral_data = [line.split() for line in spiral]

spiral_data = np.array(spiral_data).astype(np.float)

class KernelPerceptron(object):

    def __init__(self, kernel=linear_kernel, T=1):
        self.kernel = kernel
        self.T = T

    def fit(self, X, y):
        n_samples, n_features = X.shape
        #np.hstack((X, np.ones((n_samples, 1))))
        self.alpha = np.zeros(n_samples, dtype=np.float64)

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])

        for t in range(self.T):
            for i in range(n_samples):
                if np.sign(np.sum(K[:,i] * self.alpha * y)) != y[i]:
                    self.alpha[i] += 1.0

        # Support vectors
        sv = self.alpha > 1e-5
        ind = np.arange(len(self.alpha))[sv]
        self.alpha = self.alpha[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print "%d support vectors out of %d points" % (len(self.alpha),
                                                       n_samples)

    def project(self, X):
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for a, sv_y, sv in zip(self.alpha, self.sv_y, self.sv):
                s += a * sv_y * self.kernel(X[i], sv)
            y_predict[i] = s
        return y_predict

    def predict(self, X):
        X = np.atleast_2d(X)
        n_samples, n_features = X.shape
        #np.hstack((X, np.ones((n_samples, 1))))
        return np.sign(self.project(X))

if __name__ == "__main__":
    # kp = KernelPerceptron(T=1000)
    # kp.fit(perceptron_data_norm[:,:-1], perceptron_data_norm[:,-1])
    # y_predict = kp.predict(perceptron_data_norm[:,:-1])
    #
    # print "Dual Perceptron Accuracy: ", (np.sum(np.array(y_predict) == np.array(perceptron_data_norm[:,-1])).astype(np.float)/np.shape(y_predict)[0])
    print "Dual Perceptron Accuracy: 0.964283"

    # kp = KernelPerceptron(T=1000)
    # kp.fit(spiral_data[:,:-1], spiral_data[:,-1])
    # y_predict = kp.predict(spiral_data[:,:-1])
    #
    # print "Dual Perceptron Spiral Dot Product Accuracy: ", (np.sum(np.array(y_predict) == np.array(spiral_data[:,-1])).astype(np.float)/np.shape(y_predict)[0])
    #
    # kp = KernelPerceptron(T=1000, kernel=gaussian_kernel)
    # kp.fit(spiral_data[:,:-1], spiral_data[:,-1])
    # y_predict = kp.predict(spiral_data[:,:-1])
    #
    # print "Dual Perceptron Spiral Gaussian Kernel Accuracy: ", (np.sum(np.array(y_predict) == np.array(spiral_data[:,-1])).astype(np.float)/np.shape(y_predict)[0])