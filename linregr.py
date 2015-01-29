
# http://aimotion.blogspot.com/2011/10/machine-learning-with-python-linear.html

from numpy import loadtxt, zeros, ones, array, linspace, logspace
from pylab import scatter, show, title, xlabel, ylabel, plot, contour

data = loadtxt('ex1data1.txt', delimiter=',')

scatter(data[:, 0], data[:, 1], marker='o', c='b')
title('Profits distribution')
xlabel('Population of City in 10,000s')
ylabel('Profit in $10,000s')


X = data[:, 0]
y = data[:, 1]

m = y.size

it = ones(shape=(m, 2))
it[:, 1] = X

theta = zeros(shape=(2, 1))

iterations = 1500
alpha = 0.01

def compute_cost(X, y, theta):

    m = y.size

    predictions = X.dot(theta).flatten()

    sqErrors = (predictions - y) ** 2

    J = (1. / (2 * m)) * sqErrors.sum()

    return J
