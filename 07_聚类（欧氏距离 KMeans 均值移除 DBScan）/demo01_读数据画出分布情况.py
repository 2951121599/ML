import numpy as np
import matplotlib.pyplot as mp

x = np.loadtxt('../ml_data/multiple3.txt', delimiter=',')

mp.figure('K-Means Cluster', facecolor='lightgray')
mp.title('K-Means Cluster', fontsize=20)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=10)
mp.scatter(x[:, 0], x[:, 1], c='k', s=80)
mp.show()
