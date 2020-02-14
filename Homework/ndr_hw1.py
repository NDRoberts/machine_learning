"""
CS 3120 - Machine Learning - Homework 01: Gradient Descent

This code generates some sample statistical data, then uses
linear regression and gradient descent to attempt to approximate
the bias and weight values for a line-of-best-fit most closely
matching the original creation parameters.

@author: Nate Roberts
To the best of my ability, all code obtained from other sources
has been cited in comments; code not otherwise marked is my own.
Any resemblance to existing code is unintentional.

"""

import numpy as np
import matplotlib.pyplot as plt


class Dataset_1D:
    ''' A 1-dimensional dataset of input values (x) and output values(y) '''
    x_data = 0
    y_data = 0
    samples = 0

    def __init__(self, x_input, y_input):
        ''' Create new 1-dimensional dataset '''
        self.x = np.array(x_input)
        self.y = np.array(y_input)
        self.samples = len(self.x)

    def loss(self, w0, w1):
        ''' Calculate loss with a given bias and weight '''
        loss = 0
        for i in range(self.samples):
            loss = ((w1 * self.x[i] + w0) - self.y[i]) ** 2
        loss /= self.samples
        return loss

    def loss_landscape(self, w0_dims, w1_dims):
        ''' Produce a regular array of loss values, suitable for contour map '''
        # Loss landscape function based on sample code provided by Feng Jiang
        w0 = np.arange(start=w0_dims[0], stop=w0_dims[1], step=w0_dims[2])
        w1 = np.arange(start=w1_dims[0], stop=w1_dims[1], step=w1_dims[2])
        lo_la = np.zeros((len(w0), len(w1)))
        for i in range(len(w0)):
            for j in range(len(w1)):
                lo_la[j][i] = self.loss(w0[i], w1[j])
        return (w0, w1, lo_la)

    def gradient(self, w0, w1, alpha):
        ''' Determine gradient for an input pair of 'w's and update '''
        grad_w0 = 0
        grad_w1 = 0
        for i in range(self.samples):
            grad_w0 += ((w1 * self.x[i] + w0) - self.y[i]) / self.samples
            grad_w1 += ((w1 * self.x[i] + w0) - self.y[i]) * self.x[i] / self.samples
        next_w0 = w0 - alpha * grad_w0
        next_w1 = w1 - alpha * grad_w1
        return (next_w0, next_w1)

    def gradient_descent(self, with_b, with_w, alpha, max_iters):
        ''' Execute Gradient Descent to a given number of iterations '''
        # Runner function based on code by Matt Nedrich
        # https://spin.atomicobject.com/2014/06/24/gradient-descent-linear-regression/
        b = with_b
        w = with_w
        history = []
        for i in range(max_iters):
            history.append((b, w))
            b, w = self.gradient(b, w, alpha)
        history = np.array(history)
        return (b, w, history)


x_data = np.array([np.random.random() * 10 for _ in range(10)])
y_data = np.array(2 * x_data + [50 for _ in range(len(x_data))] + [np.random.random() for _ in range(len(x_data))])

x_feng = np.array([35, 38, 31, 20, 22, 25, 17, 60, 8, 60])
y_feng = 2 * x_feng + 50 + 5 * np.random.random()
data = Dataset_1D(x_feng, y_feng)
# plt.scatter(x_feng, y_feng)
# plt.show()

alpha = 0.001
max_iterations = 10000

# Contour plot display code courtesy of Feng Jiang
biases, weights, loss_map = data.loss_landscape((0, 100, 1), (-5, 5, 0.1))
plt.xlim(0, 100)
plt.ylim(-5, 5)
plt.xlabel("Bias")
plt.ylabel("Weight")
plt.contourf(biases, weights, loss_map, 50, alpha=0.5, cmap=plt.get_cmap('jet'))
plt.plot(50, 2, "x", color="orange")

b, w, trace = data.gradient_descent(0, 0, alpha, max_iterations)
plt.plot(trace[:, 0], trace[:, 1], color="white")
plt.plot(b, w, "o", color="black")
plt.show()
goodness = (b / 2) * (w / 50)
print(f"The current settings have produced a result of b = {b}, w = {w},")
print(f"which has a 'goodness-of-fit' rating of {goodness} (best fit: 1.0)")
print("(where goodness-of-fit is expressed as the product of the quotient of the final variable values and their 'true' values)")