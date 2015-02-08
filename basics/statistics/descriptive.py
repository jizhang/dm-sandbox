import math

def mean(x):
    return float(sum(x)) / len(x)

def variance(x, ddof=0):
    '''$$\sigma^{2} = E(X - \mu)^{2}$$'''
    mu = mean(x)
    cnt = len(x) - ddof
    return sum((i - mu) ** 2 for i in x) / cnt

def sd(x, ddof=0):
    '''$$\sigma$$'''
    return math.sqrt(variance(x, ddof))

def covariance(x, y, ddof=1):
    '''$$Cov(X, Y) = E[(X - \mu_{X})(Y - \mu_{Y})]$$'''
    assert len(x) == len(y)
    mu_x = mean(x)
    mu_y = mean(y)
    cnt = len(x) - ddof
    return sum((x[i] - mu_x) * (y[i] - mu_y) for i in range(len(x))) / cnt

def correlation(x, y, ddof=1):
    '''$$\rho = \frac{Cov(X, Y)}{\sigma_{X}\sigma_{Y}}$$'''
    return covariance(x, y, ddof) / sd(x, ddof) / sd(y, ddof)
