##
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/run/media/salman/Personal/Data Science/Implementations')
data = np.loadtxt('ex2data1.txt', delimiter=',')

X = data[:, 0:2]
y = data[:, 2].reshape(-1, 1)
X = np.concatenate( (np.ones((y.size, 1)), X), axis=1)
X[:10,:]
##
pos = np.where(y==1)
neg = np.where(y==0)
plt.figure()
plt.scatter(X[pos, 1], X[pos, 2], marker='x', c='r')
plt.scatter(X[neg, 1], X[neg, 2], marker='o', c='b')
##
def fit(X, y):
    from costFunction import logRegCost

    computeCost = lambda theta, X, y: logRegCost(theta, X, y)[0]
    computeGrad = lambda theta, X, y: logRegCost(theta, X, y)[1]
    initial_theta = np.zeros((X.shape[1], 1))
    from scipy.optimize import fmin_bfgs
    theta = fmin_bfgs(computeCost, initial_theta, computeGrad, args=(X, y))
    return theta
##
theta = fit(X, y)
Xplt = np.arange(20, 110, 0.01)
yplt = -theta[0]/theta[2] - theta[1]/theta[2] * Xplt
plt.figure()
plt.scatter(X[pos, 1], X[pos, 2], marker='x', c='r')
plt.scatter(X[neg, 1], X[neg, 2], marker='o', c='b')
plt.plot(Xplt, yplt, c='k')
##
# using sklearn logistic regression
from sklearn.linear_model import LogisticRegression
sklearn_theta = LogisticRegression(penalty='none', fit_intercept=False, solver='newton-cg').fit(X, y).coef_
sklearn_theta
##
