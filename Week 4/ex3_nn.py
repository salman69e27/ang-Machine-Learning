##
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/run/media/salman/Personal/Data Science/Implementations')

input_layer_size  = 400
hidden_layer_size = 25
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
theta_data = loadmat('ex3weights.mat')
Theta1 = theta_data['Theta1']
Theta2 = theta_data['Theta2']
from moduleLib import predict
pred = predict(Theta1, Theta2, X)
print( np.sum( (pred==y) )/y.size )
##
# run %matplotlib qt in your interactive console first
rp = np.random.permutation(m)
i = 0
fig = plt.figure()
ax = plt.gca()

def onclick(self):
    global i
    i += 1
    if i == m:
        plt.close()
        return
    pred = predict(Theta1, Theta2, X[rp[i], :].reshape(1, -1))[0, 0]
    plt.cla()
    displayData(X[rp[i], :].reshape(1, -1), ax)
    ax.set_title('prediction {}, digit {}. Click on picture to continue.'.format(pred, pred%10))
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('button_press_event', onclick)

pred = predict(Theta1, Theta2, X[rp[i], :].reshape(1, -1))[0, 0]
displayData(X[rp[i], :].reshape(1, -1), ax)
ax.set_title('prediction {}, digit {}. Click on picture to continue.'.format(pred, pred%10))
##
