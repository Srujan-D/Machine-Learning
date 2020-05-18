''' Linear Regression - Cost/Loss function and Gradient Descent on a dataset obtained on Kaggle. See README.md for details about the dataset '''

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

data = pd.read_csv("/home/srujan/Desktop/ML/Linear_Regression/data.csv")

def cost_fn(b, m, data):
    totalcost = 0
    for i in range(len(data)):
        x = data.iloc[i, 0]
        y = data.iloc[i, 1]
        totalcost += (y - (m * x + b)) ** 2
    return totalcost /float(len(data))

def step_gradient(bi, mi, data, learning_rate):
    dB = 0
    dM = 0
    N = float(len(data))
    for i in range(len(data)):
        x = data.iloc[i, 0]
        y = data.iloc[i, 1]
        dB += -(2/N) * (y - (np.dot(mi , x) + bi))
        dM += -(2/N) * np.dot(x , (y - (np.dot(mi , x) + bi))
    new_b = bi - (learning_rate * dB)
    new_m = mi - (learning_rate * dM)
    return [new_b, new_m]

def gradient_descent(data, b0, m0, learning_rate):
    b = b0
    m = m0
    checker = True
    while(checker):
        b_0, m_0 = b,m
        error_0 = cost_fn(b, m, data)
        [b, m] = step_gradient(b, m, data, learning_rate)
        error_new = cost_fn(b, m, data)
        if error_new > error_0:
            checker = False
    return[b_0,m_0]

learning_rate = 0.0001
initial_b = 0 # initial y-intercept guess
initial_m = 0 # initial slope guess
[b, m] = gradient_descent(data, initial_b, initial_m, learning_rate)
print("After Gradient descent b = {0}, m = {1}, error = {2}".format(b, m, cost_fn(b, m, data)))

plt.plot(data.iloc[:,0], data.iloc[:,1], '.')

axes = plt.gca()
x_values = np.array(axes.get_xlim())
y_values = b + m * x_values
plt.plot(x_values, y_values, 'r-')
