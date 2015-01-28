
# http://nbviewer.ipython.org/gist/vietjtnguyen/6655020

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import itertools
import random
import time

# color maps
cdict = {'red':   ((0.0, 0.0, 0.0),
                   (1.0, 1.0, 1.0)),
         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
         'blue':  ((0.0, 1.0, 1.0),
                   (1.0, 0.0, 0.0))}
BinaryRdBu = matplotlib.colors.LinearSegmentedColormap('BinaryRdBu', cdict, 2)
cdict = {'red':   ((0.0, 0.9, 0.9),
                   (1.0, 1.0, 1.0)),
         'green': ((0.0, 0.9, 0.9),
                   (1.0, 0.9, 0.9)),
         'blue':  ((0.0, 1.0, 1.0),
                   (1.0, 0.9, 0.9))}
LightRdBu = matplotlib.colors.LinearSegmentedColormap('LightRdBu', cdict)
cdict = {'red':   ((0.0, 1.0, 1.0),
                   (0.4, 0.7, 0.7),
                   (0.5, 0.0, 0.0),
                   (0.6, 0.7, 0.7),
                   (1.0, 1.0, 1.0)),
         'green': ((0.0, 1.0, 1.0),
                   (0.4, 0.7, 0.7),
                   (0.5, 0.0, 0.0),
                   (0.6, 0.7, 0.7),
                   (1.0, 1.0, 1.0)),
         'blue':  ((0.0, 1.0, 1.0),
                   (0.4, 0.7, 0.7),
                   (0.5, 0.0, 0.0),
                   (0.6, 0.7, 0.7),
                   (1.0, 1.0, 1.0))}
HalfContour = matplotlib.colors.LinearSegmentedColormap('HalfContour', cdict)

logistic = lambda s: 1. / (1. + np.exp(-s))

d_x = 2

phi = lambda x: x
d_z = len(phi(np.ones((d_x + 1,)))) - 1

N = 100

P_x = lambda: np.array([1.] + [np.random.uniform(-1, 1) for i in range(d_x)]) # simulates P(x)

def generate_target(d, hardness=20., offset_ratio=0.25, w_f=None):

    if w_f is None:
        w_f = np.array([np.random.uniform(-hardness * offset_ratio, hardness * offset_ratio)] +
                       [np.random.uniform(-hardness, hardness) for i in range(d)])

    f = lambda z: logistic(w_f.dot(z.T))
    P_f = lambda z: (np.array([np.random.uniform() for i in range(z.shape[0])]) <= f(z)) * 2. - 1.

    return w_f, f, P_f

w_f, f, P_f = generate_target(d_z, hardness=12.)

def generate_data_samples(N, P_x, phi, P_f):

    x = np.array([P_x() for i in range(N)])

    z = np.apply_along_axis(phi, 1, x)

    y = P_f(z)

    cross_entropy_error = lambda w: np.mean(np.log(1 + np.exp(-y * w.dot(z.T))))

    return x, z, y, cross_entropy_error

x, z, y, cross_entropy_error = generate_data_samples(N, P_x, phi, P_f)

def generate_fill_data(s=300, phi=lambda x: x):

    x_1, x_2 = np.array(np.meshgrid(np.linspace(-1, 1, s), np.linspace(-1, 1, s)))

    x_grid = np.hstack((np.ones((s * s, 1)), np.reshape(x_1, (s * s, 1)), np.reshape(x_2, (s * s, 1))))

    z_grid = np.apply_along_axis(phi, 1, x_grid)

    return x_1, x_2, x_grid, z_grid

def apply_to_fill(z_grid, func):
    s = int(np.sqrt(z_grid.shape[0]))
    return np.reshape(func(z_grid), (s, s))

x_1, x_2, x_grid, z_grid = generate_fill_data(300, phi)
f_grid = apply_to_fill(z_grid, f)

def plot_data_set_and_hypothesis(x, y, x_1, x_2, f_grid=None, title=''):
    start_time = time.time()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect(1)
    ax.set_xlabel(r'$x_1$', fontsize=18)
    ax.set_ylabel(r'$x_2$', fontsize=18)
    if not title == '':
        ax.set_title(title, fontsize=18)
    ax.xaxis.grid(color='gray', linestyle='dashed')
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.set_axisbelow(True)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.autoscale(False)

    if not f_grid is None:
        ax.pcolor(x_1, x_2, f_grid, cmap=LightRdBu, vmin=0, vmax=1)
        ax.contour(x_1, x_2, f_grid * 2 - 1, cmap=HalfContour, levels=[-0.5, 0.0, 0.5], vmin=-1, vamx=1)

    ax.scatter(x[:, 1], x[:, 2], s=40, c=y, cmap=BinaryRdBu, vmin=-1, vmax=1)

    print('Plot took {:.2f} seconds.'.format(time.time() - start_time))
    return fig

target_fig = plot_data_set_and_hypothesis(x, y, x_1, x_2, f_grid, title=r'Target, $N={:}$'.format(N))

def gradient_descent(z, y, w_h=None, eta=1.0, max_iterations=10000, epsilon=0.001):

    if w_h is None:
        w_h = np.array([0.0 for i in range(z.shape[1])])

    w_h_i = [np.copy(w_h)]

    for i in range(max_iterations):
        subset_indices = range(z.shape[0])

#         grad_E_in = np.mean(np.tile(- y[subset_indices] / ( 1.0 + np.exp(y[subset_indices] * w_h.dot(z[subset_indices].T))),
#                                     (z.shape[1], 1)
#                                     ).T * z[subset_indices], axis=0)

        numerator = -y[subset_indices]
        denominator = 1. + np.exp(y[subset_indices] * w_h.dot(z[subset_indices].T))
        tiled = np.tile(numerator / denominator, (z.shape[1], 1))

        grad_E_in = np.mean(tiled.T * z[subset_indices], axis = 0)

        w_h -= eta * grad_E_in
        w_h_i.append(np.copy(w_h))

        if np.linalg.norm(grad_E_in) <= np.linalg.norm(w_h) * epsilon:
            break

    return np.array(w_h_i)

w_h_i = gradient_descent(z, y, eta=4.0)
w_h = w_h_i[-1]
print('Number of iterations: {:}'.format(w_h_i.shape[0]))

h = lambda z: logistic(w_h.dot(z.T))
h_grid = apply_to_fill(z_grid, h)

full_N_fig = plot_data_set_and_hypothesis(x, y, x_1, x_2, h_grid, title=r'Hypothesis, $N={:}$'.format(N))

def in_sample_error(z, y, h):
    y_h = (h(z) >= 0.5) * 2 -1
    return np.sum(y != y_h) / float(len(y))

def estimate_out_of_sample_error(P_x, phi, P_f, h, N=10000, phi_h=None):
    x = np.array([P_x() for i in range(N)])
    z = np.apply_along_axis(phi, 1, x)
    if phi_h is not None:
        z_h = np.apply_along_axis(phi_h, 1, x)
    else:
        z_h = z
    y = P_f(z)
    y_h = (h(z_h) >= 0.5) * 2 -1
    return np.sum(y != y_h) / float(N)

print('Target weights: {:}'.format(w_f))
print('Hypothesis weights: {:}'.format(w_h))
print('Hypothesis in-sample error: {:.2%}'.format(in_sample_error(z, y, h)))
print('Hypothesis out-of-sample error: {:.2%}'.format(estimate_out_of_sample_error(P_x, phi, P_f, h)))

N_subset=10
subset_indices = np.random.permutation(N)[:N_subset]
x_subset = x[subset_indices, :]
z_subset = z[subset_indices, :]
y_subset = y[subset_indices]
w_h_i_subset = gradient_descent(z_subset, y_subset, eta=10.)
w_h_subset = w_h_i_subset[-1]
print('Number of iterations: {:}'.format(w_h_i_subset.shape[0]))
print('Number of iterations: {:}'.format(w_h_i_subset.shape[0]))
h_subset = lambda z: logistic(w_h_subset.dot(z.T))
h_subset_grid = apply_to_fill(z_grid, h_subset)
subset_N_fig = plot_data_set_and_hypothesis(x_subset, y_subset, x_1, x_2, h_subset_grid, title=r'Hypothesis, $N={:}$'.format(N_subset))

naked_fig = plot_data_set_and_hypothesis(x_subset, y_subset, x_1, x_2, None, title=r'Data, $N={:}$'.format(N))
