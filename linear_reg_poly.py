#!/usr/bin/python3
"""
Polynomial regression with data ingested from csv file. R-squared is computed
and printed to console. Predicted and actual data is plotted with matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
from mpl_toolkits.mplot3d import Axes3D


# ======================== READ CSV DATA FILE ==================================
# empty polynomail dataset
x = []
y = []

# open file for reading and append data to list
for line in open("data_poly.csv"):
    xn, yn = line.split(',')
    xn = float(xn.strip())
    x.append([1, xn, xn**2])
    y.append(float(yn.strip()))

# ======================== POLYNOMIAL LINEAR REGRESSION ================================
# create numpy array objects
np_x = np.array(x)
np_y = np.array(y)

# calculate weights
w = np.linalg.solve(np.dot(np_x.T, np_x), np.dot(np_x.T, np_y))
y_hat = np.dot(np_x, w)

# ======================== COMPUTE R-SQUARED ===================================
# sum of squares residual
d1 = np_y - y_hat
# sum of squares total
d2 = np_y - np_y.mean()

# compute r^2
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print(f"r-squared for predicted curve is: {round(r2,3)}")

# ======================== PLOT PREDICTED AND ACTUAL ===========================
# plot actual and predicted values
# set axis resolution
x_ticks = np.arange(0, 100, 10)
plt.xticks(x_ticks)
y_ticks = np.arange(0, 1200, 100)
plt.yticks(y_ticks)
plt.title("Polynomial Regression")
plt.xlabel("x")
plt.ylabel("y")

plt.scatter((np_x[:, 1]), np_y)
plt.plot(sorted(np_x[:, 1]), sorted(y_hat))
plt.show()
