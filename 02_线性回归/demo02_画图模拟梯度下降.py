import numpy as np
import matplotlib.pyplot as mp

train_x = np.array([0.5, 0.6, 0.8, 1.1, 1.4])
train_y = np.array([5.0, 5.5, 6.0, 6.8, 7.0])
test_x = np.array([0.45, 0.55, 1.0, 1.3, 1.5])
test_y = np.array([4.8, 5.3, 6.4, 6.9, 7.3])

times = 1000  # 定义梯度下降次数
lrate = 0.01  # 记录每次梯度下降参数变化率
epoches = []  # 记录每次梯度下降的索引(迭代的轮数)
w0, w1, losses = [1], [1], []
for i in range(1, times + 1):
    epoches.append(i)
    loss = (((w0[-1] + w1[-1] * train_x) - train_y) ** 2).sum() / 2
    losses.append(loss)
    # 通过偏导公式求两方向上的偏导数
    d0 = ((w0[-1] + w1[-1] * train_x) - train_y).sum()
    d1 = (((w0[-1] + w1[-1] * train_x) - train_y) * train_x).sum()
    print('{:4}> w0={:.8f}, w1={:.8f}, loss={:.8f}'.format(epoches[-1], w0[-1], w1[-1], losses[-1]))
    # 更新w0 与 w1
    w0.append(w0[-1] - lrate * d0)
    w1.append(w1[-1] - lrate * d1)

pred_test_y = w0[-1] + w1[-1] * test_x

# 画图
w0 = w0[:-1]
w1 = w1[:-1]
mp.figure('Linear Regression', facecolor='lightgray')
mp.title('Linear Regression', fontsize=20)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.scatter(train_x, train_y, marker='s', c='dodgerblue', alpha=0.5, s=80, label='Training')
mp.scatter(test_x, test_y, marker='D', c='orangered', alpha=0.5, s=60, label='Testing')
mp.scatter(test_x, pred_test_y, c='orangered', alpha=0.5, s=80, label='Predicted')
mp.plot(test_x, pred_test_y, '--', c='limegreen', label='Regression', linewidth=1)
mp.legend()
mp.show()
