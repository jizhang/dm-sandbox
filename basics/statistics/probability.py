import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def plot_normal(mu, sigma):
    '''http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.normal.html
    $$\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x - \mu)^{2}}{2\sigma^{2}}}$$'''
    s = np.random.normal(mu, sigma, 1000)
    count, bins, ignored = plt.hist(s, 30, normed=True)
    plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi))
                   * np.exp(-(bins - mu) ** 2 / (2 * sigma ** 2)),
             linewidth=2, color='r')
    plt.show()

def calc_normal(x, mu, sigma):
    """http://docs.scipy.org/doc/scipy/reference/tutorial/integrate.html"""
    def integrand(x, mu, sigma):
        return 1 / (np.sqrt(2 * np.pi) * sigma) \
               * np.exp(-(x - mu) ** 2 / (2 * sigma **2))
    return quad(integrand, -np.inf, x, args=(mu, sigma))[0]
