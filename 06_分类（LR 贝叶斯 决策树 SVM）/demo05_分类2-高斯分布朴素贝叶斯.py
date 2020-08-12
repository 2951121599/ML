"""
高斯分布即正态分布
"""
import numpy as np
import sklearn.naive_bayes as nb
import matplotlib.pyplot as mp

# 加载数据
data = np.loadtxt('../ml_data/multiple1.txt', unpack=False, dtype='U20', delimiter=',')
print(data.shape)  # (400, 3) 400行3列
x = np.array(data[:, :-1], dtype=float)  # 所有行的前两列
y = np.array(data[:, -1], dtype=float)  # 所有行的最后一列
print(x.shape)  # (400, 2)
print(y.shape)  # (400,)

# 创建高斯分布朴素贝叶斯分类器
model = nb.GaussianNB()
model.fit(x, y)

# 画出背景颜色 绘制分类边界线
l, r = x[:, 0].min() - 1, x[:, 0].max() + 1
b, t = x[:, 1].min() - 1, x[:, 1].max() + 1
n = 500
grid_x, grid_y = np.meshgrid(np.linspace(l, r, n), np.linspace(b, t, n))
samples = np.column_stack((grid_x.ravel(), grid_y.ravel()))
grid_z = model.predict(samples)
grid_z = grid_z.reshape(grid_x.shape)

# 散点图 在图像中绘制样本
mp.figure('Naive Bayes Classification', facecolor='lightgray')
mp.title('Naive Bayes Classification', fontsize=20)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=10)
mp.pcolormesh(grid_x, grid_y, grid_z, cmap='gray')
mp.scatter(x[:, 0], x[:, 1], c=y, cmap='brg', s=80)
mp.show()
