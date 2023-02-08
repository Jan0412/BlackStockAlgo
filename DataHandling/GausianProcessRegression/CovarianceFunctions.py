import numpy as np

def cov_SE(x, x_, sigma, length):
    return sigma ** 2 * np.exp(-(np.linalg.norm(x - x_) ** 2) / (2 * length **2))
