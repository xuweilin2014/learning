# -*- coding=utf-8 -*-
import glob
import os
import platform
import shutil
import time
import cv2
import joblib
import numpy as np

from skimage.feature import hog
from sklearn.svm import LinearSVC

# 第一个是你的类别   第二个是类别对应的名称   输出结果的时候方便查看
label_map = {1: 'cat', 2: 'chick', 3: 'snack'}

# 训练集图片的位置
train_image_path = 'image128'
# 测试集图片的位置
test_image_path = 'image128'

# 训练集标签的位置
train_label_path = os.path.join('image128','train.txt')
# 测试集标签的位置
test_label_path = os.path.join('image128','train.txt')

image_height = 128
image_width = 100

train_feat_path = 'train/'
test_feat_path = 'test/'
model_path = 'model/'


# 获得图片列表
def get_image_list(filePath, nameList):
    print('read image from ', filePath)
    img_list = []
    for name in nameList:
        img = cv2.imread(os.path.join(filePath, name))
        img_list.append(img.copy())
    return img_list


# 提取特征并保存
def get_feature(image_list, name_list, label_list, save_path):
    i = 0
    for image in image_list:

        try:
            # 如果是灰度图片  把 3 改为 -1
            image = np.reshape(image, (image_height, image_width, 3))
        except:
            print('发送了异常，图片大小 size 不满足要求：', name_list[i])
            continue

        gray = rgb2gray(image) / 255.0

        # 获取梯度方向直方图特征
        # hog(image, orientations, pixels_per_cell, cells_per_block, block_norm, transform_sqrt, feature_vector, visualise)
        # image：输入图像
        # orientations：指定 bin 的个数. scikit-image 实现的只有无符号方向. 也就是说把所有的方向都转换为 0°~180° 内, 然后按照指定的 orientation 数量划分 bins
        # pixels_per_cell：每个 cell 的像素个数，是一个 tuple 类型的数据，例如 (20, 20)
        # cells_per_block: 每个 block 内有多少个cell, tuple类型, 例如(2,2), 意思是将block均匀划分为2x2的块
        # block_num: block 内部采用的 norm 类型
        # transform_sqrt: 是否进行 power law compression, 也就是 gamma correction. 是一种图像预处理操作, 可以将较暗的区域变亮, 减少阴影和光照变化对图片的影响.
        # visualise: 是否输出 HOG image (应该是梯度图)
        fd = hog(gray, orientations=12, block_norm='L1', pixels_per_cell=[8, 8], cells_per_block=[4, 4], visualize=False, transform_sqrt=True)
        fd = np.concatenate((fd, [label_list[i]]))

        # 特征名称 = 图片名称 + '.feat'
        fd_name = name_list[i] + '.feat'
        fd_path = os.path.join(save_path, fd_name)
        # joblib 主要用来进行机器学习模型的持久化工作
        # 将 hog 特征 fd 保存到地址 fd_path 中
        joblib.dump(fd, fd_path)
        i += 1

    print("Test features are extracted and saved.")


# 变成灰度图片
def rgb2gray(im):
    gray = im[:, :, 0] * 0.2989 + im[:, :, 1] * 0.5870 + im[:, :, 2] * 0.1140
    return gray


# 获得图片名称与对应的类别标签
def get_name_label(file_path):
    print("read label from ",file_path)
    name_list = []
    label_list = []

    # 打开 train.txt 文件
    with open(file_path) as f:
        # 从 train.txt 中读取图片的名称 name 和标签 label
        for line in f.readlines():
            # 一般是 name label 三部分, 所以至少长度为 3, 所以可以通过这个忽略空白行
            if len(line) >= 3:
                name_list.append(line.split(' ')[0])
                label_list.append(line.split(' ')[1].replace('\n','').replace('\r',''))
                if not str(label_list[-1]).isdigit():
                    print("label必须为数字，得到的是：", label_list[-1], "程序终止，请检查文件")
                    exit(1)

    return name_list, label_list


# 提取特征
def extract_feature():
    # 获取训练集的图片名称和对应标签
    train_name, train_label = get_name_label(train_label_path)
    # 获取测试集的图片名称和对应标签
    test_name, test_label = get_name_label(test_label_path)

    # 获取真正的训练集图片
    train_image = get_image_list(train_image_path, train_name)
    # 获取真正的测试集图片
    test_image = get_image_list(test_image_path, test_name)

    # 将训练集的特征提取并且保存下来
    get_feature(train_image, train_name, train_label, train_feat_path)
    # 将测试集的特征提取并且保存下来
    get_feature(test_image, test_name, test_label, test_feat_path)


# 创建存放特征的文件夹
def mkdir():
    if not os.path.exists(train_feat_path):
        os.mkdir(train_feat_path)
    if not os.path.exists(test_feat_path):
        os.mkdir(test_feat_path)


# 训练和测试
def train_and_test():

    t0 = time.time()
    features = []
    labels = []
    correct_number = 0
    total = 0

    # glob.glob 用来查找符合特定规则的文件路径名。类似于文件搜索
    # 这里用来查找 'train' 目录下的所有训练集图片 hog 特征的路径
    for feat_path in glob.glob(os.path.join(train_feat_path, '*.feat')):
        data = joblib.load(feat_path)
        # data 是 hog 特征向量，最后一维保存的是标签数字
        features.append(data[:-1])
        labels.append(data[-1])

    print("Training a Linear LinearSVM Classifier.")
    clf = LinearSVC()
    clf.fit(features, labels)

    # 下面的代码是保存 svm 模型到文件中
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    joblib.dump(clf, model_path + 'model')

    # 下面的代码是加载模型, 可以注释上面的代码, 直接进行加载模型, 不进行训练
    print("训练之后的模型存放在 model 文件夹中")
    result_list = []

    # 这里用来查找 'test' 目录下的所有训练集图片 hog 特征的路径
    for feat_path in glob.glob(os.path.join(test_feat_path, '*.feat')):
        total += 1
        if platform.system() == 'Windows':
            symbol = '\\'
        else:
            symbol = '/'

        image_name = feat_path.split(symbol)[1].split('.feat')[0]
        data_test = joblib.load(feat_path)
        data_test_feat = data_test[:-1].reshape((1, -1)).astype(np.float64)
        # 对于测试集图片的 hog 向量使用 SVM 进行预测结果
        result = clf.predict(data_test_feat)
        result_list.append(image_name + ' ' + label_map[int(result[0])] + '\n')

        if int(result[0]) == int(data_test[-1]):
            correct_number += 1

    write_to_txt(result_list)
    rate = float(correct_number) / total
    t1 = time.time()

    print('准确率是： %f' % rate)
    print('耗时是 : %f' % (t1 - t0))


def write_to_txt(list):
    with open('result.txt', 'w') as f:
        f.writelines(list)
    print('每张图片的识别结果存放在result.txt里面')


if __name__ == '__main__':

    # 不存在文件夹就创建
    mkdir()

    # need_input = input('是否手动输入各个信息？y/n\n')
    # if need_input == 'y':
    #     train_image_path = input('请输入训练图片文件夹的位置,如 /home/icelee/image\n')
    #     test_image_path = input('请输入测试图片文件夹的位置,如 /home/icelee/image\n')
    #     train_label_path = input('请输入训练集合标签的位置,如 /home/icelee/train.txt\n')
    #     test_label_path = input('请输入测试集合标签的位置,如 /home/icelee/test.txt\n')
    #     size = int(input('请输入您图片的大小：如64x64，则输入64\n'))

    need_extra_feat = input('是否需要重新获取特征? y/n\n')

    # 如果需要重新获取特征的话，就删除原始的文件夹，重新获取特征并且保存到文件夹中
    if need_extra_feat == 'y':
        # shutil.rmtree 能够直接删除一个文件夹，不管里面有没有内容
        shutil.rmtree(train_feat_path)
        shutil.rmtree(test_feat_path)
        mkdir()
        # 获取特征并保存在文件夹
        extract_feature()

    # 训练并预测
    train_and_test()
