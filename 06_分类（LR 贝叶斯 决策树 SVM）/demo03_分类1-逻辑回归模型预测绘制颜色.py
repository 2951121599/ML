"""
基于逻辑回归器绘制网格化坐标颜色矩阵
将500*500个小格子放入模型 进行预测 绘制颜色
"""
import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as mp

x = np.array([
    [3, 1],
    [2, 5],
    [1, 8],
    [6, 4],
    [5, 2],
    [3, 5],
    [4, 7],
    [4, -1]])
y = np.array([0, 1, 1, 0, 0, 1, 1, 0])
# 逻辑分类器
model = lm.LogisticRegression(solver='liblinear', C=1)
model.fit(x, y)
l, r = x[:, 0].min() - 1, x[:, 0].max() + 1
b, t = x[:, 1].min() - 1, x[:, 1].max() + 1
n = 500
grid_x, grid_y = np.meshgrid(np.linspace(l, r, n), np.linspace(b, t, n))
# grid_x.ravel() 500*500 拉平为25000一行
samples = np.column_stack((grid_x.ravel(), grid_y.ravel()))  # 列合并 25000行2列

grid_z = model.predict(samples)
grid_z = grid_z.reshape(grid_x.shape)  # y再转回一维的shape
mp.figure('Logistic Classification', facecolor='lightgray')
mp.title('Logistic Classification', fontsize=20)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=10)
mp.pcolormesh(grid_x, grid_y, grid_z, cmap='gray')
mp.scatter(x[:, 0], x[:, 1], c=y, cmap='brg', s=80)
mp.show()