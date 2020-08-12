import numpy as np
import sklearn.pipeline as pl
import sklearn.preprocessing as sp
import sklearn.linear_model as lm
import matplotlib.pyplot as mp

# 采集数据
x, y = np.loadtxt('../ml_data/single.txt', delimiter=',', usecols=(0, 1), unpack=True)
x = x.reshape(-1, 1)

# 创建模型(管线)
model = pl.make_pipeline(
    sp.PolynomialFeatures(10),  # 多项式特征扩展器
    lm.LinearRegression()  # 线性回归器
)

# 训练模型
model.fit(x, y)
# 根据输入预测输出
pred_y = model.predict(x)

test_x = np.linspace(x.min(), x.max(), 1000).reshape(-1, 1)
pred_test_y = model.predict(test_x)

# 绘图
mp.figure('Polynomial Regression', facecolor='lightgray')
mp.title('Polynomial Regression', fontsize=20)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.scatter(x, y, c='dodgerblue', alpha=0.75, s=60, label='Sample')
mp.plot(test_x, pred_test_y, c='orangered', label='Regression')
mp.legend()
mp.show()
