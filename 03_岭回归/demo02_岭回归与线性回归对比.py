import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as mp
import sklearn.metrics as sm

# 采集数据
x, y = np.loadtxt('../ml_data/abnormal.txt', delimiter=',', usecols=(0, 1), unpack=True)
x = x.reshape(-1, 1)
# 创建线性回归模型
model = lm.LinearRegression()
# 训练模型
model.fit(x, y)
# 根据输入预测输出
pred_y1 = model.predict(x)

# 画图
mp.figure('Linear & Ridge', facecolor='lightgray')
mp.title('Linear & Ridge', fontsize=20)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.scatter(x, y, c='dodgerblue', alpha=0.75, s=60, label='Sample')
sorted_indices = x.T[0].argsort()
# 线性回归线
mp.plot(x[sorted_indices], pred_y1[sorted_indices], c='orangered', label='Linear')

# 岭回归
model = lm.Ridge(150, fit_intercept=True, max_iter=1000)
model.fit(x, y)
pred_y = model.predict(x)
# 评估训练结果误差
# R2得分，(0,1]区间的分值。分数越高，误差越小。
print("R2得分:", sm.r2_score(y, pred_y))
# 岭回归线
mp.plot(x, pred_y, c='green', label='Ridge Line')

mp.legend()
mp.show()
