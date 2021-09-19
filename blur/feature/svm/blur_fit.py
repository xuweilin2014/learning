from sklearn import svm
import numpy as np
import sklearn
import random
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import os

def hog_feature_fit():
    # 1.读取数据集
    path = '../data/blur'
    data = np.array([])

    for npy in os.listdir(path):
        # 读取 iris.data 文件，其中 converters 是对数据预处理的参数
        # 其实 converters 是一个字典，表示第 5 列的数据使用函数 iris_label 来进行处理
        npy_path = os.path.join(path, npy)
        npy_data = np.load(os.path.join(path, npy))

        npy_data[np.where(np.isnan(npy_data))] = 0

        if len(data) == 0:
            data = npy_data
        else:
            data = np.concatenate((data, npy_data), axis=0)

    # 2.划分数据与标签
    # indices_or_sections 若等于一个 1-D 数组，则会沿着指定的方向进行分割，1-D 数组的元素个数为 n，则数组会被分割成 n + 1 份
    x, y = data[:, :-1], data[:, -1]
    # train_test_split(train_data, train_target, test_size=0.4, random_state=0) 用于将原始数据按照比例分割为"测试集"和"训练集"
    # train_data:所要划分的样本特征集，train_target:所要划分的样本结果集，test_size:样本占比，random_state 随机数种子
    train_data, test_data, train_label, test_label = sklearn.model_selection.train_test_split(x, y, random_state=1, train_size=0.7, test_size=0.3)

    # 3.训练 SVM 分类器
    # 本例有 3 个分类，按照一对多即 1V2 内部需要创建 3 个 SVM 算法，这里设置参数即可，ovr 即一对多策略
    # kernel = 'linear' 时，为线性核，C越大分类效果越好，但有可能会过拟合
    # kernel = 'rbf' 时，为高斯核，gamma 值越小，分类界面越连续；gamma 值越大，分类界面越"散"，分类效果越好，但有可能会过拟合
    classifier = svm.SVC(C=1, kernel='linear', gamma=100)
    # ravel 的作用就是将数组维度拉成一维数组
    classifier.fit(train_data, train_label.ravel())

    # 4.计算 svc 分类器的准确率
    print('测试集准确率：', classifier.score(test_data, test_label))
    scores = cross_val_score(classifier, train_data, train_label, cv=5)
    print('交叉验证得分：{}'.format(scores))
    print('交叉验证平均得分：{:.3f}'.format(scores.mean()))

    # 5.ROC
    test_score = classifier.decision_function(test_data)
    fpr, tpr, threshold = roc_curve(test_label, test_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
    hog_feature_fit()