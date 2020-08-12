import numpy as np
import sklearn.preprocessing as sp
import sklearn.ensemble as se
import sklearn.model_selection as ms

data = np.loadtxt('../ml_data/car.txt', delimiter=',', dtype='U10')
data = data.T
encoders = []
train_x, train_y = [],[]
for row in range(len(data)):
    encoder = sp.LabelEncoder()
    if row < len(data) - 1:
        train_x.append(encoder.fit_transform(data[row]))
    else:
        train_y = encoder.fit_transform(data[row])
    encoders.append(encoder)
train_x = np.array(train_x).T
# 随机森林分类器
model = se.RandomForestClassifier(max_depth=6, n_estimators=200, random_state=7)
print(ms.cross_val_score(model, train_x, train_y, cv=4, scoring='f1_weighted').mean())  # 0.7465877061619401
model.fit(train_x, train_y)

