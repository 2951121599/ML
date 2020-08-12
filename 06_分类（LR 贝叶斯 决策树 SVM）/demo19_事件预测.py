import numpy as np
import sklearn.preprocessing as sp
import sklearn.model_selection as ms
import sklearn.svm as svm


# 自定义编码器
class DigitEncoder():
    def fit_transform(self, y):
        return y.astype(int)

    def transform(self, y):
        return y.astype(int)

    def inverse_transform(self, y):
        return y.astype(str)


# 多元分类
data = np.loadtxt('../ml_data/event.txt', delimiter=',', dtype='U10')
data = np.delete(data.T, 1, axis=0)
print(data)
encoders, x = [], []
for row in range(len(data)):
    if data[row][0].isdigit():
        # 获取自定义编码器
        encoder = DigitEncoder()
    else:
        # 获取标签编码器
        encoder = sp.LabelEncoder()
    if row < len(data) - 1:
        x.append(encoder.fit_transform(data[row]))
    else:
        y = encoder.fit_transform(data[row])
    encoders.append(encoder)
x = np.array(x).T
train_x, test_x, train_y, test_y = \
    ms.train_test_split(x, y, test_size=0.25, random_state=5)
model = svm.SVC(kernel='rbf', class_weight='balanced')
print(ms.cross_val_score(model, train_x, train_y, cv=3, scoring='accuracy').mean())
model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)
print((pred_test_y == test_y).sum() / pred_test_y.size)
data = [['Tuesday', '13:30:00', '21', '23']]
data = np.array(data).T
x = []
for row in range(len(data)):
    encoder = encoders[row]
    x.append(encoder.transform(data[row]))
x = np.array(x).T
pred_y = model.predict(x)
print(encoders[-1].inverse_transform(pred_y))
