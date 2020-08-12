"""
标签编码
"""
import numpy as np
import sklearn.preprocessing as sp

raw_samples = np.array([
    'audi', 'ford', 'audi', 'toyota', 'ford', 'bmw', 'toyota', 'ford', 'audi'])
print(raw_samples)
lbe = sp.LabelEncoder()
lbe_samples = lbe.fit_transform(raw_samples)
print(lbe_samples)  # [0 2 0 3 2 1 3 2 0]
inv_samples = lbe.inverse_transform(lbe_samples)
print(inv_samples)
# 假设得到了模型的一组预测结果
pred = np.array([0, 0, 1, 1, 3, 2, 2, 2, 3])
r = lbe.inverse_transform(pred)
print(r)
