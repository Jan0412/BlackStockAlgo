import numpy as np
from itertools import product

from  DataHandling.GausianProcessRegression.CovarianceFunctions import cov_SE

kernel_func = lambda x_, y_, sigma_f, l : sigma_f * np.exp(-(np.linalg.norm(x_ - y_) ** 2) / (2 * l ** 2))

def compute_cov_matrix(x_, x_s, sigma, sigma_n, length):
    x_ = np.asarray(x_)
    x_s = np.asarray(x_s)

    n_ = x_.shape[0]
    n_s = x_s.shape[0]

    print(n_)
    print(n_s)

    k = [cov_SE(x=i, x_=j, sigma=sigma, length=length) for (i, j) in product(x_, x_)]
    k = np.array(k).reshape((n_, n_))

    k_s = [cov_SE(x=i, x_=j, sigma=sigma, length=length) for (i, j) in product(x_s, x_)]
    k_s = np.array(k_s).reshape((n_s, n_))

    k_ss = [cov_SE(x=i, x_=j, sigma=sigma, length=length) for (i, j) in product(x_s, x_s)]
    k_ss = np.array(k_ss).reshape((n_s, n_s))

    k_ss = np.linalg.inv(k_ss)

    print(np.shape(k))
    print(np.shape(k_s))
    print(np.shape(k_ss))

    return (k + sigma_n ** 2 * np.eye(n_)), k_s, k_ss

def compute_new_mu(k, k_s, y, mode):
    k = np.asarray(k)
    k_s = np.asarray(k_s)

    y = np.asarray(y)
    mode = np.asarray(mode) if isinstance(mode, list) else np.full(y.shape[0], mode)

    return  k_s.dot(np.linalg.inv(k)).dot(y) #mode.T + k_s.dot(np.linalg.inv(k)).dot((y - mode))

def compute_new_cov(k, k_s, k_ss):
    k = np.asarray(k)
    k_s = np.asarray(k_s)
    k_ss = np.asarray(k_ss)

    return k_ss - np.dot(k_s, np.dot(np.linalg.inv(k), k_s.T)) #k_ss + k_s.dot(k.dot(k_ss.T)) #
