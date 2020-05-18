''' Linear Regression - Cost/Loss function and Gradient Descent on a dataset obtained on Kaggle. See README.md for details about the dataset '''

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

data = pd.read_csv("/kaggle/input/distancecycledvscaloriesburned/data.csv")

def cost_fn(b, m, data):
    totalcost = 0
    for i in range(len(data)):
        x = data.iloc[i, 0]
        y = data.iloc[i, 1]
        totalcost += (y - (m * x + b)) ** 2
    print ("b = ", b, " m = ", m, " avg_cost = ", totalcost /float(len(data)))   #printing the values of b, m, and the average cost helps in getting to know what is happening.
    return totalcost /float(len(data))

def step_gradient(bi, mi, data, learning_rate):
    dB = 0
    dM = 0
    N = float(len(data))
    for i in range(len(data)):
        x = data.iloc[i, 0]
        y = data.iloc[i, 1]
        dB += -(2/N) * (y - (np.dot(mi , x) + bi))
        dM += -(2/N) * np.dot(x , (y - (np.dot(mi , x) + bi)))
    bi -= (learning_rate * dB)
    mi -= (learning_rate * dM)
    return [bi, mi]

def gradient_descent(data, b0, m0, learning_rate, epochs):
    b = b0
    m = m0
    checker = True
    while(epochs):
        [b, m] = step_gradient(b, m, data, learning_rate)
        error_new = cost_fn(b, m, data)
        epochs -= 1
    return[b,m]

learning_rate = 0.0001
epochs = 250                   #number of iterations
initial_b = np.random.normal() # initial y-intercept guess
initial_m = np.random.normal() # initial slope guess
[b, m] = gradient_descent(data, initial_b, initial_m, learning_rate, epochs)
print("After Gradient descent b = {0}, m = {1}, error = {2}".format(b, m, cost_fn(b, m, data)))

plt.plot(data.iloc[:,0], data.iloc[:,1], '.')

axes = plt.gca()
x_values = np.array(axes.get_xlim())
y_values = b + m * x_values
plt.plot(x_values, y_values, 'r-')
