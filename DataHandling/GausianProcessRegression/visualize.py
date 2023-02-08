import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np

def plot_uncertainty(x, mu_matrix, cov_matrix, ax, fig, **kwargs):
    cmap = kwargs['cmap'] if 'cmap' in kwargs.keys() else 'viridis'
    alpha = kwargs['alpha'] if 'alpha' in kwargs.keys() else 0.25
    max_factor = kwargs['max_factor'] if 'max_factor' in kwargs.keys() else 5

    mu_matrix = np.asarray(mu_matrix)
    cov_matrix = np.asarray(cov_matrix)
    factors = np.linspace(start=0.25, stop=max_factor, num=(max_factor * 5))

    cm = plt.get_cmap(cmap)
    cNorm = colors.Normalize(vmin=min(factors), vmax=max(factors))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    for factor in reversed(factors):
        colorVal = scalarMap.to_rgba(factor)
        uncertainty = factor * np.sqrt(np.diag(cov_matrix))

        ax.fill_between(x=x, y1=(mu_matrix + uncertainty), y2=(mu_matrix - uncertainty), color=colorVal, alpha=alpha)

    cbar = fig.colorbar(scalarMap)
    cbar.set_label('Uncertainty Factor')

def plot_mean(x, mu_matrix, ax, **kwargs):
    alpha = kwargs['alpha'] if 'alpha' in kwargs.keys() else 1
    color =  kwargs['color'] if 'color' in kwargs.keys() else 'red'

    ax.plot(x, mu_matrix.ravel(), color=color, alpha=alpha)

def plot_multiLines(x, mu_matrix, cov_matrix, num, ax, **kwargs):
    alpha = kwargs['alpha'] if 'alpha' in kwargs.keys() else 0.25
    color =  kwargs['color'] if 'color' in kwargs.keys() else 'black'

    for _ in range(num):
        z_star = np.random.multivariate_normal(mean=mu_matrix.squeeze(), cov=cov_matrix)
        ax.plot(x, z_star, alpha=alpha, color=color)


