"""
独热编码
"""
import numpy as np
import sklearn.preprocessing as sp

raw_samples = np.array([
    [17., 100., 4000],
    [20., 80., 5000],
    [23., 75., 5500]])
# 创建独热编码器
# sparse:是否使用紧缩格式(稀疏矩阵)
ohe = sp.OneHotEncoder(sparse=False, dtype=int)
# 用独特编码器对原始样本矩阵做独热编码
r = ohe.fit_transform(raw_samples)
print(r)
