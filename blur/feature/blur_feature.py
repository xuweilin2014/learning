import numpy as np
import cv2
from numba import jit
import random
import matplotlib.pyplot as plt


# noinspection PyUnusedLocal,PyShadowingNames
def local_kurtosis(img, patch_size):
    im_height, im_width = img.shape

    # 列差分运算
    diff_x = np.diff(img, n=1, axis=1)
    tail_x = np.reshape(img[:, -1], (im_height, 1))
    dx = np.concatenate((diff_x, -tail_x), axis=1)

    # 行差分运算
    diff_y = np.diff(img, n=1, axis=0)
    tail_y = np.reshape(img[-1, :], (1, im_width))
    dy = np.concatenate((diff_y, -tail_y), axis=0)

    dx_col = im2col(dx, [patch_size, patch_size])
    dx_col = dx_col / np.sum(dx_col, axis=0)
    dy_col = im2col(dy, [patch_size, patch_size])
    dy_col = dy_col / np.sum(dy_col, axis=0)

    # 计算 (BX) ^ 2 和 (BY) ^ 2
    norm_x_square = (dx_col - np.mean(dx_col, axis=0)) ** 2
    norm_y_square = (dy_col - np.mean(dy_col, axis=0)) ** 2

    # 计算 K(BX) 和 K(BY)
    qx = np.mean(norm_x_square ** 2, axis=0) / (np.mean(norm_x_square, axis=0) ** 2)
    qy = np.mean(norm_y_square ** 2, axis=0) / (np.mean(norm_y_square, axis=0) ** 2)

    # 将 qx 和 qy 中值为 nan 的数设置为 qx 或者 qy 中的最小值
    qx[np.isnan(qx).astype('int')] = np.min(qx[(~np.isnan(qx)).astype('int')])
    qy[np.isnan(qy).astype('int')] = np.min(qy[(~np.isnan(qy)).astype('int')])

    q = np.min((qx, qy), axis=0)

    q = list(map(lambda x: x + random.randint(0, 1), q))
    plot_hist(q, xlabel='Value of kurtosis', ylabel='Number of pixels', title='Kurtosis Feature Response', bins=5000)

    return q

# noinspection PyUnusedLocal,PyShadowingNames
@jit(cache=True)
def gradient_histogram_span(img, patch_size):
    im_height, im_width = img.shape

    # 列差分运算
    diff_x = np.diff(img, n=1, axis=1)
    dx = np.concatenate((diff_x, -np.reshape(img[:, -1], (im_height, 1))), axis=1)
    # 行差分运算
    diff_y = np.diff(img, n=1, axis=0)
    dy = np.concatenate((diff_y, -np.reshape(img[-1, :], (1, im_width))), axis=0)
    # 计算图像的幅度值
    magnitude = np.sqrt(dx ** 2 + dy ** 2)
    offset = (patch_size - 1) / 2

    mag_col = im2col(magnitude, [patch_size, patch_size])
    mag_col_dup = np.concatenate((mag_col, mag_col), axis=0)

    num = mag_col.shape[1]
    sigma = np.zeros((1, num))

    for i in range(num):
        # 使用两个高斯模型来进行建模，并且获取两个高斯模型的方差
        v1, v2 = em_gmm(mag_col_dup[:, i])
        s1, s2 = np.sqrt(v1), np.sqrt(v2)
        # 选取较小的标准差作为特征
        sigma[0][i] = np.min([s1, s2])

    return sigma

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

def local_auto_correlation(img):
    return img


# noinspection PyShadowingNames
def im2col(img, patch_size):
    img_shape = img.shape
    sh = img_shape[0] - patch_size[0] + 1
    sw = img_shape[1] - patch_size[1] + 1
    # 设 A 为 m * n，如果划分为 [p, q] 的块，最后的矩阵的行数为 p * q，列数为 (m-p+1)*(n-q+1)
    result = np.empty((patch_size[0] * patch_size[1], sh * sw))
    for i in range(sw):
        for j in range(sh):
            result[:, i * sh + j] = img[j:j + patch_size[0], i:i + patch_size[1]].ravel(order='F')

    return result

def gradient_distribution(img):
    im_height, im_width = img.shape

    # 列差分运算
    diff_x = np.diff(img, n=1, axis=1)
    dx = np.concatenate((diff_x, -np.reshape(img[:, -1], (im_height, 1))), axis=1)
    # 行差分运算
    diff_y = np.diff(img, n=1, axis=0)
    dy = np.concatenate((diff_y, -np.reshape(img[-1, :], (1, im_width))), axis=0)
    # 计算图像的幅度值
    gradient = np.concatenate((dx, dy), axis=0)
    gradient = np.reshape(gradient, (gradient.shape[0] * gradient.shape[1]))

    plot_hist(gradient, xlabel='Gradient', ylabel='Number of pixels', title='Gradient Distribution', bins=5000)

def magnitude_distribution(img):
    im_height, im_width = img.shape

    # 列差分运算
    diff_x = np.diff(img, n=1, axis=1)
    dx = np.concatenate((diff_x, -np.reshape(img[:, -1], (im_height, 1))), axis=1)
    # 行差分运算
    diff_y = np.diff(img, n=1, axis=0)
    dy = np.concatenate((diff_y, -np.reshape(img[-1, :], (1, im_width))), axis=0)
    # 计算图像的幅度值
    magnitude = np.sqrt(dx ** 2 + dy ** 2)
    magnitude = np.concatenate((magnitude, -magnitude), axis=0)
    magnitude = np.reshape(magnitude, (magnitude.shape[0] * magnitude.shape[1]))

    plot_hist(magnitude, xlabel='Gradient Magnitude', ylabel='Number of pixels', title='Gradient Magnitude Distribution', bins=5000)

    v1, v2 = em_gmm(magnitude)
    s1, s2 = np.sqrt(v1), np.sqrt(v2)
    mu = 0

    x = np.linspace(-1, 1, 100)
    gauss1 = np.exp(-(x - mu) ** 2 / (2 * s1 ** 2)) / (np.sqrt(2 * np.pi) * s1)
    gauss2 = np.exp(-(x - mu) ** 2 / (2 * s2 ** 2)) / (np.sqrt(2 * np.pi) * s2)

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
    img_path = 'race_60.jpg'
    image = cv2.imread(img_path, 0)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    patch_size = 11

    magnitude_distribution(image)

    # f1 = local_kurtosis(image, patch_size)
    # f2 = gradient_histogram_span(image, patch_size)
    # f3 = local_auto_correlation(image)