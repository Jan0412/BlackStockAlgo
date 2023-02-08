#imports
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from GaussianProcess import *
from visualize import *


def test():
    # Generate data-samples
    n = 100
    x = np.linspace(start=0, stop=1, num=n)
    f = lambda val: np.sin((4 * np.pi) * val) + np.sin((7 * np.pi) * val) + 5.5 * val + 2
    f_x = f(val=x)

    # generate points
    sigma_n = 0.2
    epsilon = np.random.normal(loc=0, scale=sigma_n, size=n)
    y = f_x + epsilon

    # plot generated points
    fig, ax = plt.subplots()
    sns.scatterplot(x=x, y=y, label='sample points', ax=ax)
    sns.lineplot(x=x, y=f(val=x), color='red', label='f(x)', ax=ax)

    # Gaussian Process Regression (GPR)
    X = np.linspace(0, 1, 500)
    k, k_s, k_ss = compute_cov_matrix(x_=x, x_s=X, sigma=1, sigma_n=1, length=1)
    f_mu_s = compute_new_mu(k=k, k_s=k_s, y=y)
    f_cov_s = compute_new_cov(k=k, k_s=k_s, k_ss=k_ss)
    f_cov_s = np.absolute(f_cov_s)

    print(f_mu_s)
    print(f_mu_s.shape)

    print(f_cov_s)
    print(f_cov_s.shape)

    print(np.sqrt(np.diag(f_cov_s)))

    plot_mean(x=X, mu_matrix=f_mu_s, color="green", alpha=1, ax=ax)
    plot_uncertainty(x=X, mu_matrix=f_mu_s, cov_matrix=f_cov_s, ax=ax, fig=fig, alpha=0.5)
    plot_multiLines(x=X, mu_matrix=f_mu_s, cov_matrix=f_cov_s, num=50, ax=ax, alpha=0.1)

    ax.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    test()