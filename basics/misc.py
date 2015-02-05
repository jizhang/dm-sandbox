import math

def C(n, k):
    return P(n, k) / math.factorial(k)

def P(n, k):
    return math.factorial(n) / math.factorial(n - k)
