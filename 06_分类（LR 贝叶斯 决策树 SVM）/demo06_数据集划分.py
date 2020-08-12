import numpy as np
import sklearn.model_selection as ms
import sklearn.naive_bayes as nb
import matplotlib.pyplot as mp

data = np.loadtxt('../ml_data/multiple1.txt', unpack=False, dtype='U20', delimiter=',')
print(data.shape)
x = np.array(data[:, :-1], dtype=float)
y = np.array(data[:, -1], dtype=float)
# 划分训练集和测试集
train_x, test_x, train_y, test_y = \
    ms.train_test_split(x, y, test_size=0.25, random_state=7)
# 朴素贝叶斯分类器
model = nb.GaussianNB()
# 用训练集训练模型
model.fit(train_x, train_y)

l, r = x[:, 0].min() - 1, x[:, 0].max() + 1
b, t = x[:, 1].min() - 1, x[:, 1].max() + 1
n = 500
grid_x, grid_y = np.meshgrid(np.linspace(l, r, n), np.linspace(b, t, n))
samples = np.column_stack((grid_x.ravel(), grid_y.ravel()))
grid_z = model.predict(samples)
grid_z = grid_z.reshape(grid_x.shape)

pred_test_y = model.predict(test_x)
# 计算并打印预测输出的精确度
print((test_y == pred_test_y).sum() / pred_test_y.size)

mp.figure('Naive Bayes Classification', facecolor='lightgray')
mp.title('Naive Bayes Classification', fontsize=20)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=10)
mp.pcolormesh(grid_x, grid_y, grid_z, cmap='gray')
mp.scatter(test_x[:, 0], test_x[:, 1], c=test_y, cmap='brg', s=80)  # 画出测试集
mp.show()
