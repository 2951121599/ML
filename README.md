# 机器学习

## 监督学习

### 回归模型

- 线性回归

  - 原理: 样本-预测函数-损失函数-梯度下降求最小值
  - 调用API: 创建线性回归模型 执行fit和predict
  - 评估: R2得分

- 岭回归

  - 原理: 防止特殊样本带跑偏, 加入了正则项 正则项为0时,即为线性回归

- 多项式回归

  - 一元多项式回归可以看做多元线性回归  进行了特征扩展
  - 实现步骤: 1.转为多元线性方程, 给出多项式最高次数 2.将w1,w2..当做特征,交给线性回归器去做训练

- 决策树回归

  - 原理: 相似的输入产生相似的输出 (例如: 预测薪资或者波士顿房价)
  - 副产物: 特征重要性
  - 步骤: 选第一个特征进行子表划分, 使每个子表中特征值相同. 按同样步骤划分子表, 直到所有特征全部使用完, 得到叶级子表           预测时.逐一匹配, 直到找到与之匹配的叶级子表, 通过求平均值做回归业务

- 集合算法

  - 思想: 三个臭皮匠顶个诸葛亮
  - 步骤: 构建多棵不同的决策树模型
  - 包括正向激励和随机森林

    - 正向激励: 开始时为样本随机分配权重, 之后对那些预测值与实际值不准的样本, 提高其权重
    - 随机森林: 随机选择部分样本而且随机选择部分特征 (即随机选择行和列)         好处: 1. 规避了强势样本对预测结果的影响   2. 削弱了强势特征的影响  3. 使模型的预测能力更加泛化

### 分类模型

- 逻辑分类

  - 逻辑回归 (二分类)

    - sigmoid函数将连续数据离散化

  - 多元分类

    - 通过多个二元分类器解决多元分类问题

- 贝叶斯分类

  - 原理: 根据统计概率实现分类   条件独立, 特征值之间没有因果关系
  - 公式: P(A|B) = P(B|A)*P(A) / P(B)
  - 高斯分布朴素贝叶斯

- 决策树分类

  - 步骤: 使用随机森林分类器进行训练 交叉验证输出f1得分

- SVM

  - 原理

    - 1.寻求最优分类边界
    - 2.基于核函数的升维变换

      - 通过核函数的特征变换,增加新特征,使得低维度空间中的线性不可分问题变为高纬度空间中的线性可分问题

        - 线性核函数  linear

          - C

        - 多项式核函数  poly  高次方幂

          - C degree

        - 径向基核函数  rbf  符合正态分布

          - C  gamma

  - 适用于小型数据集 大型数据集用决策树
  - 缺点: 样本数据量大时,会扩展特征,样本空间大,算法变复杂,模型变复杂,性能会变慢

- 选取模型、调参与评估

  - 样本类别均衡化

    - 原理: 通过类别权重均衡化, 使所占比例较小的样本权重较高, 所占比例较大的样本权重较低
    - 方法: 上下采样(多的砍)  数据增强(少的增)

  - 调参

    - 验证曲线

      - 调节超参数

    - 学习曲线

      - 调节训练集大小

    - 网格搜索

      - 寻求最优超参数组合

  - 评估手段

    - 数据集划分

      - 对于分类问题,训练集和测试集应该在每个类别样本中抽取特定的百分比

    - 交叉验证

      - 把样本空间中的所有样本均分成N份,使用不同的训练集训练模型,对不同的测试集进行测试时输出指标得分
      - 指标

        - 精确度(正确/总数)
        - 查准率(准不准)
        - 召回率(够不够)
        - f1得分

    - 混淆矩阵

      - 行代表实际类别 列代表预测类别

    - 分类报告

      - 得到混淆矩阵和交叉验证的查准率 召回率 f1得分  方便分析出哪些样本是异常样本

  - 置信概率

    - 根据样本与分类边界的距离远近,对其预测类别的可信程度进行量化. 离边界越近,置信概率越低

## 无监督学习

### 聚类

- 欧氏距离

  - 用两个样本对应的特征值之差的平方和的平方根,来表示两个样本的相似性

- K-Means算法

  - 步骤: 1.随机选择K个样本作为聚类中心, 计算每个样本到各个聚类中心的欧氏距离, 将该样本分配到与之聚类中心最近的聚类中心所在的类别里     2.根据第一步所得的聚类划分, 分别计算每个聚类的几何中心, 将几何中心作为新的聚类中心, 重复第一步, 直到计算所得的几何中心和聚类中心重合或接近重合为止
  - 注意: 1.聚类数K必须事先已知, 借助某些评估指标, 优选最好的聚类数       2.聚类中心的初始选择会影响到最终聚类划分的结果, 初始中心尽量选择距离较远的样本

- 均值漂移

  - 服从某种概率分布规则,使用不同的概率密度函数拟合样本中的统计直方图, 不断移动密度函数的中心,直到获得最佳拟合效果为止
  - 特点: 1.聚类数不必事先已知  2. 聚类划分的结果相对稳定  3. 样本空间应该服从某种概率分布规则

- DBSCAN

  - 步骤: 从样本空间中选一样本, 以事先给定的半径作圆,凡被该圆圈中的样本都视为相同的聚类, 以这些被圈中的样本为圆心继续做圆,重复以上过程, 不断扩大被圈中的样本, 直到没有新的样本加入为止, 至此得到一个聚类. 于剩下样本中, 重复以上过程, 直至耗尽样本
  - 借助轮廓系数, 优选最优半径  区间为 [-1,1], 其中-1代表效果差, 0代表聚类重叠, 1代表分类效果好
  - 特点: 1.事先给定的半径会影响最后的聚类效果,可以借助轮廓系数选择较优方案    2.样本分为三类: 外周样本 孤立样本  核心样本

### 降维

- PCA主成分分析

  - 降维例子:房子的长,宽,面积,房间数量 就可以去掉长宽两个维度

- 优点

  - 从高维压缩到的低维中最大程度地保留了数据的信息
  - 数据的可视化

## 强化学习

### 奖励和惩罚

## 其它

### 推荐系统

## 总结

### 回归模型

- 线性回归 岭回归 多项式回归 决策树 正向激励 随机森林
- 评估: R2得分 计算预测误差

### 分类模型

- 逻辑回归(二分类 sigmoid) 
  朴素贝叶斯(高斯分布) 
  决策树(相似的输入产生相似的输出) 
  SVM(对特征扩展性能慢 小数据集)
- 模型评估

  - 1.混淆矩阵 
    2.分类报告(准确度 查准率 召回率 f1得分)

- 选择模型

  - 数据集的划分 交叉验证 验证曲线 学习曲线 网格搜索

### 聚类模型

- 欧氏距离 KMeans算法 均值漂移 DBScan

### 降维模型

- PCA主成分分析(线性降维)  SVD奇异值分解

### 拿到一组样本后

- 1.观察属于回归问题还是分类问题
- 2.针对数据集进行初步分析(每一个特征值的离散型与连续性, 及其数值分布)
- 3.选择合适的模型
- 4.评估模型

### 面临问题

- 建模
- 评估
- 优化

## 数据预处理

### 均值移除

- 均值为1 标准差为0

### 范围缩放

- 特征值的范围缩放 0-1

### 归一化

- 正则化 数值不重要 占比重要

### 二值化

- 图像边缘检测

### 独热编码

- 稀疏矩阵

### 标签编码

- 字符串转为数字

## 一般过程

### 数据处理

- 数据收集
- 数据清洗
- 特征工程

### 机器学习

- 选择
- 训练
- 评估
- 测试

### 业务运维

- 应用模型
- 维护模型
