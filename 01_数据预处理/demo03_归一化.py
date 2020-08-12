"""
数据样本预处理
返回归一化预处理后的样本矩阵
sp.normalize(array, norm='l1')
"""
import numpy as np
import sklearn.preprocessing as sp

samples = np.array([
    [20., 10., 5.],
    [10., 5., 2.],
    [18., 12., 13.],
])
# l1范数，向量中个元素绝对值之和
r = sp.normalize(samples, norm='l1')
print(r)
print(r.sum(axis=1))  # 列之和

# l2范数，向量中个元素平方之和
r = sp.normalize(samples, norm="l2")
print(r)
print((r ** 2).sum(axis=1))
