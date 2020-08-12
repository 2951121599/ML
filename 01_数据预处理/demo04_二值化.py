"""
数据样本预处理
二值化
sp.normalize(array, norm='l1')
"""
import numpy as np
import sklearn.preprocessing as sp
import matplotlib.pyplot as mp

samples = np.array([
    [17., 100., 4000],
    [20., 80., 5000],
    [23., 75., 5500]
])
# l1范数，向量中个元素绝对值之和
bin = sp.Binarizer(threshold=80)
r = bin.fit_transform(samples)
print(r)

# 加载图像 二值化处理
import scipy.misc as sm

# 灰度图像
img = sm.imread('../ml_data/lily.jpg', True)
print(img.shape)
mp.subplot(121)  # 画子图 一行两列第一幅
mp.imshow(img, cmap='gray')

#  二值化图像
bin = sp.Binarizer(threshold=127)
img2 = bin.fit_transform(img)
mp.subplot(122)  # 画子图 一行两列第二幅
mp.imshow(img2, cmap='gray')

# 抠图操作
img3 = sm.imread('../ml_data/lily.jpg')  # 彩色图
img4 = np.zeros_like(img3)  # 空图片
img4[img2 == 1] = img3[img2 == 1]  # 掩码覆盖上去即可
mp.figure('抠图操作')
mp.imshow(img4)
mp.show()
