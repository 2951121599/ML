"""
数据样本预处理
范围缩放
"""
import numpy as np
import sklearn.preprocessing as sp

raw_samples = np.array([
    [17., 100., 4000],
    [20., 80., 5000],
    [23., 75., 5500]])
print(raw_samples)

# 传统方法实现
mms_samples = raw_samples.copy()
for col in mms_samples.T:
    col_min = col.min()
    col_max = col.max()
    a = np.array([
        [col_min, 1],
        [col_max, 1]])
    b = np.array([0, 1])
    x = np.linalg.solve(a, b)  # 求解  x[0]为k  x[1]为b
    col = x[0] * col + x[1]  # y = kx+b
print(mms_samples)

# API实现
# 根据给定范围创建一个范围缩放器
mms = sp.MinMaxScaler(feature_range=(0, 1))
# 用范围缩放器实现特征值的范围缩放
mms_samples = mms.fit_transform(raw_samples)
print(mms_samples)
