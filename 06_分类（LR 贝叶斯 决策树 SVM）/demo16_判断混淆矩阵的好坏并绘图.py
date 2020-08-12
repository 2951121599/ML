"""
判断混淆矩阵的好坏并绘图
"""
import numpy as np
import matplotlib.pyplot as mp

data = [[22, 0, 0, 0],
        [0, 27, 1, 0],
        [0, 0, 25, 0],
        [0, 0, 0, 25]]

data = np.array(data)
print(data)

mp.figure('title', facecolor='lightgray')
mp.title('title', fontsize=20)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=10)
mp.pcolormesh(data, cmap='gray')
mp.show()
