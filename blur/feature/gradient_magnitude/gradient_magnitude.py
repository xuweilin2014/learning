import cv2
import numpy as np
from auto_correlation.local_auto_correlation import *
from numba import jit
import matplotlib.pyplot as plt

# noinspection PyUnusedLocal,PyShadowingNames
"""
计算第二个特征，默认长度为 128 维。假如有一个 121 * 121 的图像，使用一个大小为 11 * 11 的滑动窗口，类似于 matlab 中
im2col 函数的 sliding 模式，因此最终会生成 12321 （111 * 111）个 patch，对每一个 patch 计算出其梯度幅度向量，
然后使用混合高斯模型 GMM 对其进行拟合，GMM模型中只有两个分量，然后我们选择其中较大的标准差作为代表当前 patch 的特征，
也就是每个 patch 最终会有一个特征值，最后把所有 patch 中的特征值汇总起来，就可以得到一个 12321 维的特征向量 sigma，
然后根据 sigma 向量计算出直方图 hist(sigma)（bins 默认为 128），直方图的纵坐标就是像素的数量，然后进行归一化，
最后得到的 128 维的特征 feature = hist(sigma)
"""
def gradient_histogram_span(img_path, patch_size, bins=512):
    # 如果是模糊图像，那么使用红色
    if img_path.find('blur') != -1:
        color = 'red'
    # 如果是清晰图像，那么使用蓝色
    else:
        color = 'blue'

    img = cv2.imread(img_path, 0)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    im_height, im_width = img.shape

    # 列差分运算，求出图像在 x 方向的梯度 Ix
    diff_x = np.diff(img, n=1, axis=1)
    dx = np.concatenate((diff_x, -np.reshape(img[:, -1], (im_height, 1))), axis=1)
    # 行差分运算，求出图像在 y 方向的梯度 Iy
    diff_y = np.diff(img, n=1, axis=0)
    dy = np.concatenate((diff_y, -np.reshape(img[-1, :], (1, im_width))), axis=0)
    # 计算图像的梯度幅度值，mag = sqrt(Ix ^ 2 + Iy ^ 2)
    magnitude = np.sqrt(dx ** 2 + dy ** 2)
    offset = (patch_size - 1) / 2

    # 类似于 matlab 中的 im2col 函数，这里实现的是其 sliding 模式，即滑动窗口模式，
    # mag_col 矩阵中的每一列表示的是每一个 patch 中的梯度向量
    mag_col = im2col(magnitude, [patch_size, patch_size])
    mag_col_dup = np.concatenate((mag_col, mag_col), axis=0)

    num = mag_col.shape[1]
    sigma = np.zeros(num)

    for i in range(num):
        # 使用两个高斯模型来对每一个 patch 中的梯度向量进行建模，并且获取两个高斯模型的方差
        v1, v2 = em_gmm(mag_col_dup[:, i])
        s1, s2 = np.sqrt(v1), np.sqrt(v2)
        # 选取较大的标准差作为特征
        sigma[i] = np.max([s1, s2])

    # 正常显示负号
    plt.rcParams['axes.unicode_minus'] = False
    # feature = hist(sigma)
    feature, _, _ = plt.hist(sigma, bins=bins, facecolor=color, range=(0, 0.5), edgecolor=color, alpha=0.7)
    # 对特征进行标准化
    feature /= np.sum(feature)
    # 显示横轴标签
    plt.xlabel('Value of sigma')
    # 显示纵轴标签
    plt.ylabel('Number of pixels')
    # 显示图标题
    plt.title('GMM Model Variance')
    plt.show()

    return feature

@jit(cache=True)
def em_gmm(data):
    """
    compute expectation for GMM with two Gaussian Component where mean are fixed as zero.
    Our goal is mainly to compute the variance for two Gaussian.
    :param data: input data
    :return: Guassian variances
    """

    """Initialize parameters"""
    # v1 表示第一个高斯模型的方差 (不是标准差)
    v1 = 0.5
    # v2 表示第二个高斯模型的方差
    v2 = 0.0001
    # pi 表示第一个高斯模型的权重系数，第二个高斯模型的权重系数直接 1 - pi
    pi = 0.5
    N = len(data)

    pi_init = 2 * pi
    max_iter = 100
    counter = 0

    # EM 算法的收敛条件
    while abs(pi_init - pi) / pi_init > 1e-3 and counter < max_iter:
        counter += 1
        pi_init = pi

        # x_square 相当于 x 的平方
        x_square = data ** 2
        """ Expectation Step """
        w_gaussian = pi * gaussian_distribution(x_square, v2)
        # 计算第 2 个高斯模型对各个观测数据的响应度，因此 1 - gamma 为第 1 个高斯模型的响应度
        gamma = w_gaussian / ((1 - pi) * gaussian_distribution(x_square, v1) + w_gaussian)

        """ Maximization Step """
        # 分别更新两个高斯模型的方差
        v1 = np.sum((1 - gamma) * x_square) / np.sum(1 - gamma)
        v2 = np.sum(gamma * x_square) / np.sum(gamma)
        # 更新第 2 个高斯模型的权重值
        pi = np.sum(gamma) / N

    return v1, v2

def gaussian_distribution(x_square, sigma):
    """
    计算高斯函数的分布
    :param x_square: x_square 表示的是观测数据的平方，这里默认均值为 0
    :param sigma: sigma 表示的是高斯函数的方差
    """
    return np.exp(-x_square / (2 * sigma)) / np.sqrt(2 * np.pi * sigma)

def gradient_magnitude_contrast(img_path, img_blur_path, bins=128, patch_size=11):
    hist_no_blur = gradient_histogram_span(img_path, patch_size, bins=bins)
    hist_blur = gradient_histogram_span(img_blur_path, patch_size, bins=bins)

    x = np.linspace(0, 0.5, bins)
    plt.plot(x, hist_blur, label='blur', color='red', marker='o', linestyle='-')
    plt.plot(x, hist_no_blur, label='no blur', color='blue', marker='o', linestyle='-')
    plt.legend()
    plt.show()


"""
计算单个图像的梯度幅值向量 mag，然后使用包含两个分量的高斯模型来对 mag 进行拟合，求出每一个高斯模型分量的方差 v，
而均值默认为 0，再绘制出这两个高斯模型的对比图，同时，会绘制出 mag 向量的直方图
"""
def magnitude_distribution(img_path):
    img = cv2.imread(img_path, 0)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    im_height, im_width = img.shape

    # 列差分运算，也就是计算图像在 x 方向上的梯度 Ix
    diff_x = np.diff(img, n=1, axis=1)
    dx = np.concatenate((diff_x, -np.reshape(img[:, -1], (im_height, 1))), axis=1)
    # 行差分运算，也就是计算图像在 y 方向上的梯度 Iy
    diff_y = np.diff(img, n=1, axis=0)
    dy = np.concatenate((diff_y, -np.reshape(img[-1, :], (1, im_width))), axis=0)
    # 计算图像的幅度值，mag(I) = sqrt(Ix^2 + Iy^2)
    magnitude = np.sqrt(dx ** 2 + dy ** 2)
    magnitude = np.concatenate((magnitude, -magnitude), axis=0)
    # 将梯度幅值矩阵 mag 转换成一个一维的向量，方便后续计算幅值向量的直方图 hist(mag)
    magnitude = np.reshape(magnitude, (magnitude.shape[0] * magnitude.shape[1]))

    plot_hist(magnitude, xlabel='Gradient Magnitude', ylabel='Number of pixels', title='Gradient Magnitude Distribution', bins=5000)

    # 使用混合高斯模型（只有两个分量）来对这 mag 梯度幅值向量进行拟合
    # 然后返回这两个高斯模型的方差（默认使用的高斯模型的均值为 0）v1, v2
    v1, v2 = em_gmm(magnitude)
    s1, s2 = np.sqrt(v1), np.sqrt(v2)
    # 所有高斯模型的均值默认都为 0
    mu = 0

    x = np.linspace(-1, 1, 100)
    gauss1 = np.exp(-(x - mu) ** 2 / (2 * s1 ** 2)) / (np.sqrt(2 * np.pi) * s1)
    gauss2 = np.exp(-(x - mu) ** 2 / (2 * s2 ** 2)) / (np.sqrt(2 * np.pi) * s2)

    # 绘制出两个高斯模型的折线图
    plt.plot(x, gauss1, label='sigma = ' + str(round(s1, 3)), c='deepskyblue')
    plt.plot(x, gauss2, label='sigma = ' + str(round(s2, 3)), c='r')
    plt.legend()
    plt.show()

def plot_hist(data, xlabel='x label', ylabel='y label', title='title', cn=False, bins=1000):
    if cn:
        # 用黑体显示中文
        plt.rcParams['font.sans-serif'] = ['SimHei']

    # 正常显示负号
    plt.rcParams['axes.unicode_minus'] = False

    plt.hist(data, bins=bins, facecolor="blue", range=(-1, 1), edgecolor="red", alpha=0.7)
    # 显示横轴标签
    plt.xlabel(xlabel)
    # 显示纵轴标签
    plt.ylabel(ylabel)
    # 显示图标题
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    gradient_magnitude_contrast('../input/376.jpg', '../input/376_blur.jpg')
    gradient_magnitude_contrast('../input/386.jpg', '../input/386_blur.jpg')
    gradient_magnitude_contrast('../input/389.jpg', '../input/389_blur.jpg')
    gradient_magnitude_contrast('../input/445.jpg', '../input/445_blur.jpg')
    gradient_magnitude_contrast('../input/449.jpg', '../input/449_blur.jpg')
    gradient_magnitude_contrast('../input/499.jpg', '../input/499_blur.jpg')