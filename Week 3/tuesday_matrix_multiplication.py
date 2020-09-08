# tuesday_matrix_multiplication.py
#   Matrix math.
# Devon Gardner
# 09/08/2020

import matplotlib.pyplot as plt
import numpy as np

W = np.array([[0.0411, -1.03, 6.72, -3.81, 30.4]]).T

# Count 0 to 11 in a (12,1) column vector:
months = np.arange(12) # just like range()
months = months.reshape((12,1))
ones = np.ones((12,1)) # shape = tuple

# Horizontal stack: each param = 1 column
X = np.hstack((months**4, months**3, months**2, months, ones))

# Matrix multiplication
Y_pred = X @ W
print(Y_pred)