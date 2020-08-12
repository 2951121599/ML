import numpy as np
import sklearn.cluster as sc
import sklearn.metrics as sm
import matplotlib.pyplot as mp

x = np.loadtxt('../ml_data/perf.txt', delimiter=',')
epsilons, scores, models = np.linspace(0.3, 1.2, 10), [], []
for epsilon in epsilons:
    # DBSCAN聚类器
    model = sc.DBSCAN(eps=epsilon, min_samples=5)
    model.fit(x)
    # 借助轮廓系数优选最优半径
    score = sm.silhouette_score(
        x, model.labels_, sample_size=len(x), metric='euclidean')
    scores.append(score)
    models.append(model)
scores = np.array(scores)
best_index = scores.argmax()
best_epsilon = epsilons[best_index]
print("best_epsilon:", best_epsilon)
best_score = scores[best_index]
print("best_score:", best_score)
best_model = models[best_index]
pred_y = best_model.fit_predict(x)
core_mask = np.zeros(len(x), dtype=bool)
core_mask[best_model.core_sample_indices_] = True
offset_mask = best_model.labels_ == -1
periphery_mask = ~(core_mask | offset_mask)
mp.figure('DBSCAN Cluster', facecolor='lightgray')
mp.title('DBSCAN Cluster', fontsize=20)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=10)
labels = best_model.labels_
# 绘制所有核心样本的索引下标
mp.scatter(x[core_mask][:, 0], x[core_mask][:, 1], c=labels[core_mask],
           cmap='brg', s=80, label='Core')
# 绘制所有外周点样本
mp.scatter(x[periphery_mask][:, 0], x[periphery_mask][:, 1], alpha=0.5,
           c=labels[periphery_mask], cmap='brg', marker='s', s=80, label='Periphery')
# 绘制所有孤立点样本
mp.scatter(x[offset_mask][:, 0], x[offset_mask][:, 1],
           c=labels[offset_mask], cmap='brg', marker='x', s=80, label='Offset')
mp.legend()
mp.show()
