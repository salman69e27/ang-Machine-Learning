## 
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../Shared Library')
data = np.loadtxt('ex1data1.txt', delimiter=',') #load data
data
##
plt.figure()
plt.scatter(data[:, 0], data[:, 1], marker='x', c='r')
##
X = data[:, 0].reshape(-1, 1)
y = data[:, 1]
m = len(y)
##
X = np.concatenate( (np.ones((m, 1)), X), axis = 1) #include bias column
initial_theta = np.zeros(2) #initialize theta
iterations = 25000
alpha = 0.01 #gradient descent learning rate
##
def fit(X, y, theta, alpha, num_iters):
    import costFunction
    import importlib
    importlib.reload(costFunction)
    from costFunction import linRegCost
    history = np.zeros(num_iters)
    for i in range(num_iters):
        J, grad = linRegCost(theta, X, y)
        history[i] = J
        theta = theta - alpha*grad
    return theta, history
##
theta, history = fit(X, y, initial_theta, alpha, iterations)
plt.plot(X[:, 1], X@theta, c='b')
##
from costFunction import linRegCost
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
J_vals = np.array(
        [ [linRegCost(np.array([theta0, theta1]), X, y)[0] for theta1 in theta1_vals]
            for theta0 in theta0_vals])
J_vals = J_vals.T
print(J_vals.shape)
plt.figure()
plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20)) #plot cost function contours
plt.scatter(theta[0], theta[1], color='black')
##
# using sklearn
from sklearn.linear_model import LinearRegression
sklearn_theta = LinearRegression(fit_intercept=False).fit(X, y).coef_.reshape(-1, 1)
print(theta)
print(sklearn_theta)
plt.figure()
plt.plot(X[:, 1], X@theta, c='b')
plt.plot(X[:, 1], X@sklearn_theta, c='g')
plt.scatter(X[:, 1], y, marker='x', c='r')
##
