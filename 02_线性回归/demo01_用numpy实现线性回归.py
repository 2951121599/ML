import numpy as np
import matplotlib.pyplot as mp

train_x = np.array([0.5, 0.6, 0.8, 1.1, 1.4])
train_y = np.array([5.0, 5.5, 6.0, 6.8, 7.0])

# 基于梯度下降理论 找到最优w0与w1 使得loss取最小值
w0 = 1
w1 = 1
times = 1000
lrate = 0.01
for i in range(times):
    # 通过偏导公式求两方向上的偏导数
    d0 = (w0 + w1 * train_x - train_y).sum()
    d1 = (train_x * (w0 + w1 * train_x - train_y)).sum()
    # 更新w0与w1
    w0 = w0 - lrate * d0
    w1 = w1 - lrate * d1
print(w0, w1)

# 画图
mp.figure('Linear Regression', facecolor='lightgray')
mp.title('Linear Regression', fontsize=16)
mp.grid(linestyle=':')
mp.scatter(train_x, train_y, s=80, color='dodgerblue', label='Samples')
pred_y = w1 * train_x + w0
mp.plot(train_x, pred_y, linewidth=2, color='orangered', label='Regression Lion')
mp.legend()
mp.show()
