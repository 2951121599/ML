import pickle

import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as mp
import sklearn.metrics as sm

# 采集数据
x, y = np.loadtxt('../ml_data/abnormal.txt', delimiter=',', usecols=(0, 1), unpack=True)
x = x.reshape(-1, 1)  # -1代表转换后reshape之后的n行
# 创建模型
model = lm.LinearRegression()  # 线性回归
# 训练模型
model.fit(x, y)
# 根据输入预测输出
pred_y = model.predict(x)

# 评估训练结果误差
# R2得分，(0,1]区间的分值。分数越高，误差越小。
print("R2得分:", sm.r2_score(y, pred_y))

# 显示散点
mp.figure('Linear Regression', facecolor='lightgray')
mp.title('Linear Regression', fontsize=20)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.scatter(x, y, c='dodgerblue', alpha=0.75, s=60, label='Sample')
mp.plot(x, pred_y, c='orangered', label='Regression')

mp.legend()
mp.show()
