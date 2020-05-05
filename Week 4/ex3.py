##
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/run/media/salman/Personal/Data Science/Implementations')

input_layer_size  = 400
num_labels = 10
from scipy.io import loadmat
data = loadmat('ex3data1.mat')
X = data['X']
y = data['y']
##
m = y.size
rand_indices = np.random.randint(0, m, 100)
from moduleLib import displayData
sel = X[rand_indices, :]
displayData(sel)
##
theta_t = np.array( [ [-2], [-1], [1], [2] ] )
X_t = np.concatenate( (np.ones( (5, 1) ), np.arange(1, 16).reshape(3, 5).T/10), axis=1 )
y_t = np.array([ [1], [0], [1], [0], [1] ])
lambda_t = 3
from costFunction import regLogRegCost
J, grad = regLogRegCost(theta_t, X_t, y_t, lambda_t)
print(J, '\n', grad)
##
lmbda = 0.1
from moduleLib import oneVsAll
all_theta = oneVsAll(X, y, num_labels, lmbda)
##
from moduleLib import predictOneVsAll
pred = predictOneVsAll(all_theta, X)
print(np.mean((pred==y)))
##
