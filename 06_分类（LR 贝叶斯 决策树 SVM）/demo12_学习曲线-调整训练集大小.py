"""
学习曲线：模型性能 = f(训练集大小)
"""
import numpy as np
import sklearn.preprocessing as sp
import sklearn.ensemble as se
import sklearn.model_selection as ms

data = np.loadtxt('../ml_data/car.txt', delimiter=',', dtype='U10')
data = data.T
encoders = []
train_x, train_y = [], []
for row in range(len(data)):
    encoder = sp.LabelEncoder()
    if row < len(data) - 1:
        train_x.append(encoder.fit_transform(data[row]))
    else:
        train_y = encoder.fit_transform(data[row])
    encoders.append(encoder)
train_x = np.array(train_x).T
# 随机森林分类器
model = se.RandomForestClassifier(max_depth=9, n_estimators=135, random_state=7)

# 学习曲线
params = np.linspace(0.1, 1.0, 10)
_, train_scores, test_scores = ms.learning_curve(model, train_x, train_y, train_sizes=params, cv=5)
# 按行的方向求均值
score = test_scores.mean(axis=1)
# print("测试集得分:", score)  # 10个平均分数

# 绘制验证曲线
import matplotlib.pyplot as mp

mp.figure("Learning Curve", facecolor="lightgray")
mp.title("Learning Curve", fontSize=16)
mp.xlabel("train size")
mp.ylabel("f1")
mp.grid(linestyle=':')
mp.plot(params, score, 'o-', color='dodgerblue', label='Learning Curve')
mp.legend()
mp.show()
# 训练模型
model.fit(train_x, train_y)

# 预测
data = [
    ['high', 'med', '5more', '4', 'big', 'low', 'unacc'],
    ['high', 'high', '4', '4', 'med', 'med', 'acc'],
    ['low', 'low', '2', '4', 'small', 'high', 'good'],
    ['low', 'med', '3', '4', 'med', 'high', 'vgood']]

data = np.array(data).T
test_x, test_y = [], []
for row in range(len(data)):
    encoder = encoders[row]
    if row < len(data) - 1:
        test_x.append(encoder.transform(data[row]))
    else:
        test_y = encoder.transform(data[row])
test_x = np.array(test_x).T
pred_test_y = model.predict(test_x)
print(encoders[-1].inverse_transform(test_y))
print(encoders[-1].inverse_transform(pred_test_y))
print((pred_test_y == test_y).sum() / pred_test_y.size)  # 得分
