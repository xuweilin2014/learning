from sklearn import svm
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib


# 在分类中类别标签必须为数字，所以应该将 iris KCF 中的第 5 列的类别通过转换变为数字
def iris_label(s):
    iris_dict = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return iris_dict[s]


# 1.读取数据集
path = 'iris.data'
# 读取 iris.KCF 文件，其中 converters 是对数据预处理的参数
# 其实 converters 是一个字典，表示第 5 列的数据使用函数 iris_label 来进行处理
data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_label})

# 2.划分数据与标签
# indices_or_sections 若等于一个 1-D 数组，则会沿着指定的方向进行分割，1-D 数组的元素个数为 n，则数组会被分割成 n + 1 份
x, y = np.split(data, indices_or_sections=(4,), axis=1)
# 为了数据可视化的方便，只使用 x 数据中的前两维
x = x[:, 0:2]
# train_test_split(train_data, train_target, test_size=0.4, random_state=0) 用于将原始数据按照比例分割为"测试集"和"训练集"
# train_data:所要划分的样本特征集，train_target:所要划分的样本结果集，test_size:样本占比，random_state 随机数种子
train_data, test_data, train_label, test_label = sklearn.model_selection.train_test_split(x, y, random_state=1, train_size=0.6, test_size=0.4)

# 3.训练 SVM 分类器
# 本例有 3 个分类，按照一对多即 1V2 内部需要创建 3 个 SVM 算法，这里设置参数即可，ovr 即一对多策略
# kernel = 'linear' 时，为线性核，C越大分类效果越好，但有可能会过拟合
# kernel = 'rbf' 时，为高斯核，gamma 值越小，分类界面越连续；gamma 值越大，分类界面越"散"，分类效果越好，但有可能会过拟合
classifier = svm.SVC(C=1, kernel='rbf', gamma=10, decision_function_shape='ovr')
# ravel 的作用就是将数组维度拉成一维数组
classifier.fit(train_data, train_label.ravel())

# 4.计算 svc 分类器的准确率
# 方法 1
print('训练集: ', classifier.score(train_data, train_label))
print('测试集：', classifier.score(test_data, test_label))

# 方法 2
pre_train_label = classifier.predict(train_data)
pre_test_label = classifier.predict(test_data)
print('训练集: ', accuracy_score(train_label, pre_train_label))
print('测试集: ', accuracy_score(test_label, pre_test_label))

# 5.查看决策函数，返回的是样本到分类超平面的距离
# 若选用 ovr，则每个样本会产生 n 个距离值（n 为类别种类数）
# 若选用 ovo，则每个样本会产生 n * (n - 1) / 2 个距离值
decisions = classifier.decision_function(train_data)
print('train_decision_function: ', decisions)
predicts = classifier.predict(train_data)
print(predicts)

# 6.绘制图形
# 确定坐标轴范围
# 第 0 维特征的范围，分别取最小值和最大值
x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
# 第 1 维特征的范围，分别取最小值和最大值
x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
# mgrid 返回多维结构，np.mgrid[第1维, 第2维, 第3维, ...]
# 第 n 维的书写方式为：a:b:c，c 表示步长，为实数表示间隔，该维长度为 [a,b)，左开右闭; cj 为复数则表示点数；改长度为 [a,b]
# np.mgrid[min:max:3j, min:max:5j]，表示生成一个 3*5 的矩阵，也就是说 x 坐标有 3 个点，y 坐标有 5 个点
x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
# .flat 将数组转换为一维，并且返回这个一维数组的迭代器
# numpy.stack() function is used to join a sequence of same dimension arrays along a new axis.
grid_test = np.stack((x1.flat, x2.flat), axis=1)
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
cm_light = matplotlib.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = matplotlib.colors.ListedColormap(['g', 'r', 'b'])

plt.xlabel('iris length')
plt.ylabel('iris width')
# 测试样本点的显示，这里显示的是测试样本点的 ground-truth
plt.subplot(211)
plt.scatter(test_data[:, 0], test_data[:, 1], c=test_label[:, 0], s=30, cmap=cm_dark)
# 测试样本点的预测值显示
plt.subplot(212)
plt.scatter(test_data[:, 0], test_data[:, 1], c=pre_test_label, s=30, cmap=cm_dark)

# 限制 x, y 轴的边界
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.show()
