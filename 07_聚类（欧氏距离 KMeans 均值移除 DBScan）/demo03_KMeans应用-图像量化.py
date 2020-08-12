"""
KMeans 聚类算法应用到图像量化领域
把一张图像上所包含的颜色值进行聚类划分,
求每一类别的均值后再生成新的图像,可以达到图像降维的目的
图像的轮廓保留也较好
"""
import scipy.misc as sm
import sklearn.cluster as sc
import matplotlib.pyplot as mp


# 通过K均值聚类量化图像中的颜色
def quant(image, n_clusters):
    """
    :param image: 图像
    :param n_clusters: 聚类数量
    :return:
    """
    x = image.reshape(-1, 1)  # n行1列
    model = sc.KMeans(n_clusters=n_clusters)
    model.fit(x)
    y = model.labels_
    centers = model.cluster_centers_.ravel()
    return centers[y].reshape(image.shape)


original = sm.imread('../ml_data/lily.jpg', True)
quant2 = quant(original, 2)
quant3 = quant(original, 3)
quant4 = quant(original, 4)
mp.figure('Image Quant', facecolor='lightgray')
mp.subplot(221)
mp.title('Original', fontsize=16)
mp.axis('off')
mp.imshow(original, cmap='gray')

mp.subplot(222)
mp.title('Quant-2', fontsize=16)
mp.axis('off')
mp.imshow(quant2, cmap='gray')
mp.subplot(223)
mp.title('Quant-3', fontsize=16)
mp.axis('off')
mp.imshow(quant3, cmap='gray')
mp.subplot(224)
mp.title('Quant-4', fontsize=16)
mp.axis('off')
mp.imshow(quant4, cmap='gray')
mp.tight_layout()
mp.show()
