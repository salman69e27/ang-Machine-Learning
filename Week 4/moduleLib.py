import numpy as np
import matplotlib.pyplot as plt
from seaborn import heatmap

def displayData(X, ax=None):
    from math import ceil, sqrt, floor
    if ax == None:
        plt.figure()
        ax = plt.gca()
    example_width = ceil(sqrt(X.shape[1]))
    m, n = X.shape
    example_height = int(n/example_width)

    num_row = floor(sqrt(m))
    num_col = ceil(m/num_row)

    pad = 1
    display_array = - np.ones((pad+num_row * (example_height+pad),
            pad+num_col * (example_width+pad) ))
    cur_ex = 0
    for j in range(num_row):
        for i in range(num_col):
            if cur_ex == m:
                break
            max_val = np.max( np.abs(X[cur_ex, :]) )
            display_array[ pad + j * (example_height + pad) + np.arange(0, example_height),
                    (pad + i * (example_width + pad) + np.arange(0, example_width)).reshape(-1, 1)] = ( 
                            X[cur_ex, :].reshape(example_height, example_width)/max_val)
            cur_ex += 1
        if cur_ex==m:
            break

    heatmap(display_array, cmap='Greys', cbar=False)
    plt.gca().set_yticks([])
    plt.gca().set_xticks([])

def oneVsAll(X, y, num_labels, lmbda):
    m, n = X.shape
    initial_theta = np.zeros( (n+1, 1) )
    Theta = np.ones( (num_labels, n+1) )
    X = np.concatenate( (np.ones( (m, 1) ), X), axis=1 )

    from costFunction import regLogRegCost
    from scipy.optimize import fmin_bfgs
    computeCost = lambda theta, X, y, lmbda: regLogRegCost(theta, X, y, lmbda)[0]
    computeGrad = lambda theta, X, y, lmbda: regLogRegCost(theta, X, y, lmbda)[1]

    for i in range(num_labels):
        Theta[i, :] = fmin_bfgs(computeCost, initial_theta, computeGrad, 
                args=(X, (y==i+1)*1, lmbda), maxiter=50)
    return Theta

def predictOneVsAll(Theta, X):
    X = np.concatenate( ( np.ones( (X.shape[0], 1) ), X ), axis=1 )
    predictions = X@Theta.T
    predictedLabels = predictions.argmax(axis=1)+1
    return predictedLabels.reshape(-1, 1)

def predict(Theta1, Theta2, X):
    sigmoid = lambda z: 1/(1+np.exp(-z))
    X = np.concatenate( (np.ones( (X.shape[0], 1) ), X), axis=1 )
    A2 = sigmoid(X@Theta1.T)  
    A2 = np.concatenate( (np.ones((A2.shape[0], 1)), A2), axis=1)
    A3 = sigmoid(A2@Theta2.T)
    return (A3.argmax(axis=1)+1).reshape(-1, 1)
