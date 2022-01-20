#!/usr/bin/python3
"""
1D linear regression with data ingested from csv file. R-squared is computed
and printed to console. Predicted and actual data is plotted with matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
import csv


# ======================== READ CSV DATA FILE ==================================
# empty linear dataset
x = []
y = []

# open file for reading
f = open("1D_dataset.csv")

# csv file read object
csv_f = csv.reader(f)

# for row in csv file append column data to x and y lists
for row in csv_f:
    x.append(float(row[0]))
    y.append(float(row[1]))


# ======================== 1D LINEAR REGRESSION ================================
# create numpy array objects
np_x = np.array(x)
np_y = np.array(y)

# find slope and intercept a and b for minimum error (see 1D linear regression derivation)
# common demoninator for a and b
denom = np_x.dot(np_x) - np_x.mean()*np_x.sum()

# compute a and b NOTE: the below expressions are obtained by dividing the mathematical
# solution numerator and denominator by N (see 1D linear regression derivation)
a = (np_x.dot(np_y) - np_y.mean()*np_x.sum())/denom
b = (np_y.mean()*np_x.dot(np_x) - np_x.mean()*np_x.dot(np_y))/denom

# predicted y values
y_hat = a*np_x + b


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
x_ticks = np.arange(0, 200, 10)
plt.xticks(x_ticks)
y_ticks = np.arange(0, 200, 10)
plt.yticks(y_ticks)

# plot data
plt.scatter(np_x, np_y)
plt.plot(np_x, y_hat)
plt.show()
