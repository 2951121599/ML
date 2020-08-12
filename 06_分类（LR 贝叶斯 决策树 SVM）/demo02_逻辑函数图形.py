"""
绘制函数y = 1 / (1 + np.exp(-x))的图像
"""
import numpy as np
import matplotlib.pyplot as mp

x = np.linspace(-100, 100)
y = 1 / (1 + np.exp(-x))
mp.plot(x, y, linestyle=':')
mp.show()
