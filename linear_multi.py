#!/usr/bin/python3
"""
Multiple linear regression with data ingested from csv file. R-squared is computed
and printed to console. Predicted and actual data is plotted with matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
from mpl_toolkits.mplot3d import Axes3D

# ======================== READ CSV DATA FILE ==================================
# empty linear dataset
x = []
y = []

# open file for reading and append data to list
for line in open("2D_dataset.csv"):
    x1, x2, y1 = line.split(',')
    x.append([1, float(x1), float(x2)])
    y.append(float(y1))

# ======================== MULTIPLE LINEAR REGRESSION ================================
# create numpy array objects
np_x = np.array(x)
np_y = np.array(y)

# calculate weights
# numpy default * does element by element operations
# to do matrix muliiplaction we use np.dot
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
# set axes
ax = plt.axes(projection='3d')
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y")

# rotate axes (elevation, azimuth) deg
ax.view_init(15, -45)

# print scatter and line plot
ax.scatter(np_x[:, 1], np_x[:, 1], np_y)
ax.plot(sorted(np_x[:, 1]), sorted(np_x[:, 1]), sorted(y_hat))
plt.show()
