"""
基于决策树的正向激励回归器模型
对于预测值与实际值误差大的样本提高其权重 构建更多的决策树
预测波士顿地区房屋价格
"""
import sklearn.datasets as sd
import sklearn.utils as su
import sklearn.tree as st
import sklearn.metrics as sm
import sklearn.ensemble as se

# 加载波士顿地区房价数据集
boston = sd.load_boston()

# |CRIM|ZN|INDUS|CHAS|NOX|RM|AGE|DIS|RAD|TAX|PTRATIO|B|LSTAT|
# 犯罪率|住宅用地比例|商业用地比例|是否靠河|空气质量|房间数|年限|距中心区距离|路网密度|房产税|师生比|黑人比例|低地位人口比例|

# 打乱原始数据集的输入和输出
x, y = su.shuffle(boston.data, boston.target, random_state=7)

# 划分训练集和测试集
train_size = int(len(x) * 0.8)
train_x, test_x, train_y, test_y = \
    x[:train_size], \
    x[train_size:], \
    y[:train_size], \
    y[train_size:]

# 创建决策树回归模型
model = st.DecisionTreeRegressor(max_depth=4)
# 训练模型
model.fit(train_x, train_y)
# 测试模型
pred_test_y = model.predict(test_x)
print("决策树回归模型得分:", sm.r2_score(test_y, pred_test_y))  # 0.8202560889408635

# 创建基于决策树的正向激励回归器模型
model = se.AdaBoostRegressor(model, n_estimators=400, random_state=7)
# 训练模型
model.fit(train_x, train_y)
# 测试模型
pred_test_y = model.predict(test_x)
print("正向激励回归器模型得分:", sm.r2_score(test_y, pred_test_y))  # 0.9068598725149652
