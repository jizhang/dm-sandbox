import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

def hist(x):
    # x = np.random.normal(200, 25, 10000)
    n, bins, patches = plt.hist(x, 50, normed=True)
    y = mlab.normpdf(bins, np.mean(x), np.std(x))
    plt.plot(bins, y, 'k--', linewidth=1.5)
    plt.show()
