"""
数据样本预处理
均值移除
"""
import numpy as np
import sklearn.preprocessing as sp
raw_samples = np.array([
    [17., 100., 4000],
    [20., 80., 4500],
    [23., 95., 6000]])

# scale函数用于对函数进行预处理，实现均值移除
r = sp.scale(raw_samples)
print(r)
# 第一列均值为0
print(r.mean(axis=0))
# 第一列标准差为1
print(r.std(axis=0))