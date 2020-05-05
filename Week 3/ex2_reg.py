##
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/run/media/salman/Personal/Data Science/Implementations')

data = np.loadtxt('ex2data2.txt', delimiter=',')
X = data[:, :-1]
y = data[:, -1]
##
def make_poly(X, degree):
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    return poly.fit_transform(X)
X = make_poly(X, 6)
##
pos = np.where(y==1)
neg = np.where(y==0)
plt.scatter(X[pos, 1], X[pos, 2], marker='x', c='r')
plt.scatter(X[neg, 1], X[neg, 2], marker='o', c='b')
##
def fit(X, y):
    from costFunction import regLogRegCost

    computeCost = lambda theta, X, y, lmbda: regLogRegCost(theta, X, y, lmbda)[0]
    computeGrad = lambda theta, X, y, lmbda: regLogRegCost(theta, X, y, lmbda)[1]
    initial_theta = np.zeros((X.shape[1], 1))
    lmbda = 1
    from scipy.optimize import fmin_bfgs
    theta = fmin_bfgs(computeCost, initial_theta, computeGrad, args=(X, y, lmbda))
    return theta
##
theta = fit(X, y)
theta = theta.reshape(-1, 1)
plt.figure()
pos = np.where(y==1)
neg = np.where(y==0)
plt.scatter(X[pos, 1], X[pos, 2], marker='x', c='r')
plt.scatter(X[neg, 1], X[neg, 2], marker='o', c='b')
u = np.linspace(-1, 1.5, 50)
v = u.copy()
z = np.zeros((u.size, v.size))
for i in range(u.size):
    for j in range(v.size):
        z[i, j] = make_poly( np.array( [[u[i], u[j]]] ), 6 )@theta
plt.contour(u, v, z.T, [0])
##
