import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from GaussianProcessRegression import GaussianProcess, visualize

# Generate data-samples
#n = 100
#x = np.linspace(start=-1, stop=1, num=n)
#f = lambda x_ : np.sin((4*np.pi)*x_) + np.sin((7*np.pi)*x_) - 5.5 * x_ + 2
#f_x = f(x_=x)

# generate points
#sigma_n = 0.2
#epsilon = np.random.normal(loc=0, scale=sigma_n, size=n)
#y = f_x + epsilon

X = np.arange(-5, 5, 0.2)
x = np.array([-4, -3, -2, -1, 1])
y = np.sin(x)

# plot generated points
fig, ax = plt.subplots()
#sns.scatterplot(x=x, y=y, label='sample points', ax=ax)
#sns.lineplot(x=x, y=f(x_=x), color='red', label='f(x)', ax=ax)

# Gaussian Process Regression (GPR)
k, k_s, k_ss = GaussianProcess.compute_cov_matrix(x_=x, x_s=X, sigma=1, sigma_n=1, length=1)
f_mu_s = GaussianProcess.compute_new_mu(k=k, k_s=k_s, y=y, mode=0)
f_cov_s = GaussianProcess.compute_new_cov(k=k, k_s=k_s, k_ss=k_ss)

print(f_mu_s)
print(f_mu_s.shape)

print(f_cov_s)
print(f_cov_s.shape)

print(np.sqrt(np.diag(f_cov_s)))

visualize.plot_mean(x=X, mu_matrix=f_mu_s, color="green", alpha=1, ax=ax)
visualize.plot_uncertainty(x=X, mu_matrix=f_mu_s, cov_matrix=f_cov_s, ax=ax, fig=fig, alpha=1)
visualize.plot_multiLines(x=X, mu_matrix=f_mu_s, cov_matrix=f_cov_s, num=50, ax=ax, alpha=0.1)

ax.legend(loc='upper right')
plt.show()