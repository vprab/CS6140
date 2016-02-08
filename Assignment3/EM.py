__author__ = 'Vishant'
import numpy as np
import math
from scipy.stats import multivariate_normal

#### LOADING DATA ####

with open("2gaussian.txt") as gaussian_2:
    gaussian_data2 = [line.split() for line in gaussian_2]

gaussian_2_data = np.array(gaussian_data2).astype(np.float)

with open("3gaussian.txt") as gaussian_3:
    gaussian_3_data = [line.split() for line in gaussian_3]

gaussian_3_data = np.array(gaussian_3_data).astype(np.float)

#### EM GAUSSIAN ####


def multivariate_gaussian_density(x, mu, sigma):
    n = np.shape(x)[1]
    sigma_norm = np.linalg.norm(sigma)
    sigma_pinv = np.linalg.pinv(sigma)

    return 1/(((2*np.pi) ** (n/2.0)) * max(math.sqrt(sigma_norm), 1e-30)) * math.exp(-1/2 * ((x - mu) * sigma_pinv * (x - mu).T)[0,0])


def em_2dim_gaussians(data, num_gaussians, num_iterations):
    m, n = np.shape(data)
    ws = np.random.rand(num_gaussians)
    means = np.random.rand(num_gaussians, 2)
    sigma = np.cov(data.T)
    covs = np.random.rand(num_gaussians, 2, 2)

    for i in range(num_gaussians):
        covs[i] = sigma

    for c in range(num_iterations):
        prob_y_from_j = np.zeros((m, num_gaussians))
        for i, record in enumerate(data):
            for j in range(num_gaussians):
                prob_y_from_j[i, j] = ws[j]*multivariate_normal.pdf(np.matrix(record), means[j], covs[j], allow_singular=True)

        prob_y_from_j /= np.sum(np.matrix(prob_y_from_j), axis=1)
        prob_for_j = np.sum(prob_y_from_j, axis=0)

        for j in range(num_gaussians):
            prob_j = prob_for_j[j]

            ws[j] = prob_j/m

            means[j] = np.zeros(shape=(1, 2))
            for i in range(m):
                means[j] += 1/prob_j * prob_y_from_j[i, j] * data[i]

            covs[j] = np.zeros(shape=(2,2))
            for i in range(m):
                diff = np.matrix(data[i] - means[j])
                covs[j] += 1/prob_j * prob_y_from_j[i, j] * np.dot(diff.T, diff)

    return ws, means, covs

print em_2dim_gaussians(gaussian_2_data, 2, 100)
print em_2dim_gaussians(gaussian_3_data, 3, 100)
