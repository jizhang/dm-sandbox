import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
print a.dot(b) / np.linalg.norm(a) / np.linalg.norm(b)
