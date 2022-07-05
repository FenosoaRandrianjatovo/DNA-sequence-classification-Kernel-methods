import numpy as np


def sigma_from_median(X):
    pairwise_diff = X[:, :, None] - X[:, :, None].T
    pairwise_diff *= pairwise_diff
    euclidean_dist = np.sqrt(pairwise_diff.sum(axis=1))
    return np.median(euclidean_dist)

def linear_kernel(x1, x2):
    return x1 @ x2.T

def polynomial_kernel(X1, X2, power=2):
    return np.power((1 + linear_kernel(X1, X2)),power)

def rbf_kernel(X1, X2, sigma=10):
    X2_norm = np.sum(X2 ** 2, axis = -1)
    X1_norm = np.sum(X1 ** 2, axis = -1)
    gamma = 1 / (2 * sigma ** 2)
    K = np.exp(- gamma * (X1_norm[:, None] + X2_norm[None, :] - 2 * X1 @ X2.T))
    return K
