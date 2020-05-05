import numpy as np
def linRegCost(theta, X, y):
    '''
    X: feature np array
    y: labels np array (vector)
    theta: coef np array (vector)
    '''
    m = y.size
    prediction = X@theta
    J = np.sum( (prediction-y)**2 )/2/m
    grad = X.T@(prediction-y)/m
    return J, grad.flatten()

def logRegCost(theta, X, y):
    sigmoid = lambda z : 1/(1+np.exp(-z))
    theta = theta.reshape(-1, 1)
    y = y.reshape(-1, 1)
    m = y.size
    prediction = sigmoid(X@theta)
    J = ( -y.T @ np.log(prediction) - (1-y).T @ np.log(1-prediction) )/m
    grad = X.T@(prediction-y)/m
    return J[0, 0], grad.flatten()

def regLogRegCost(theta, X, y, lmbda):
    sigmoid = lambda z : 1/(1+np.exp(-z))
    theta = theta.reshape(-1, 1)
    y = y.reshape(-1, 1)
    m = y.size
    prediction = sigmoid(X@theta)
    J = ( ( -y.T @ np.log(prediction) - (1-y).T @ np.log(1-prediction) )/m 
            + lmbda/2/m * (theta[1:, :].T@theta[1:, :])[0, 0] )
    grad = X.T@(prediction-y)/m 
    grad[1:, :] += lmbda/m * theta[1:, :]
    return J[0, 0], grad.flatten()
