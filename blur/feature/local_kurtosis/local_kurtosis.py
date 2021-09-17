import cv2
import matplotlib.pyplot as plt
import numpy as np
import warnings
from auto_correlation.local_auto_correlation import im2col

warnings.filterwarnings("ignore", category=RuntimeWarning)

"""
生成第一个特征，默认长度为 128 维。使用的公式 eq-1 为：min(ln(K(Bx) + 3), ln(K(By) + 3))，
其中 Bx 和 By 分别表示的是模糊图像 (B) 在 x 方向和 y 方向上的梯度矩阵，K(X) 使用峰度值计算公式 eq-2（详见论文），
并且，计算的方式为，假如有一个 121 * 121 的图像，使用一个大小为 11 * 11 的滑动窗口，然后类似于 matlab 中
im2col 函数的 sliding 模式，因此最终会生成 12321 （111 * 111）个 patch，对每一个 patch 使用 eq-1 计算
出一个值，最后会生成一个 12321 维的 q 向量，然后根据 q 向量计算出直方图 hist(q)（bins 默认为 128），直方图的纵坐标
就是像素的数量，然后进行归一化，得到的 128 维的特征 hist(q)
"""
# noinspection PyUnusedLocal,PyShadowingNames,PySimplifyBooleanCheck
def local_kurtosis(img, patch_size, blur=False, bins=512, display=True):
    # 如果是模糊图像，那么使用红色
    if blur == True:
        color = 'red'
    # 如果是清晰图像，那么使用蓝色
    else:
        color = 'blue'

    im_height, im_width = img.shape
    # 列差分运算
    diff_x = np.diff(img, n=1, axis=1)
    tail_x = np.reshape(img[:, -1], (im_height, 1))
    dx = np.concatenate((diff_x, -tail_x), axis=1)

    # 行差分运算
    diff_y = np.diff(img, n=1, axis=0)
    tail_y = np.reshape(img[-1, :], (1, im_width))
    dy = np.concatenate((diff_y, -tail_y), axis=0)

    # 类似于 matlab 中的 im2col 函数，这里实现的是其 sliding 模式，即滑动窗口模式
    dx_col = im2col(dx, [patch_size, patch_size])
    dx_col = dx_col / np.sum(dx_col, axis=0)

    dy_col = im2col(dy, [patch_size, patch_size])
    dy_col = dy_col / np.sum(dy_col, axis=0)

    # 计算 (Bx) ^ 2 和 (By) ^ 2
    norm_x_square = (dx_col - np.mean(dx_col, axis=0)) ** 2
    norm_y_square = (dy_col - np.mean(dy_col, axis=0)) ** 2

    # 计算 K(Bx) 和 K(By)
    qx = np.mean(norm_x_square ** 2, axis=0) / (np.mean(norm_x_square, axis=0) ** 2)
    qy = np.mean(norm_y_square ** 2, axis=0) / (np.mean(norm_y_square, axis=0) ** 2)

    # 将 qx 和 qy 中值为 nan 的数设置为 qx 或者 qy 中的最小值
    qx[np.isnan(qx).astype('int')] = np.min(qx[(~np.isnan(qx)).astype('int')])
    qy[np.isnan(qy).astype('int')] = np.min(qy[(~np.isnan(qy)).astype('int')])

    # 分别选择 qx 和 qy 中的最小值，组成一个新的向量
    q = np.min((qx, qy), axis=0)
    # feature = hist(q)，也就是根据得到的 q 向量计算出直方图，然后把直方图的值作为第一个特征向量
    feature, _, _ = plt.hist(q, bins=bins, facecolor=color, range=(0, 18), edgecolor=color, alpha=0.7)
    feature /= np.sum(feature)

    if display:
        # 显示横轴标签
        plt.xlabel('Value of kurtosis')
        # 显示纵轴标签
        plt.ylabel('Number of pixels')
        # 显示图标题
        plt.title('Kurtosis Feature Response')
        plt.show()

    feature[np.isnan(feature).astype('int')] = np.min(feature[(~np.isnan(feature)).astype('int')])
    return feature

"""
用来绘制模糊图像和清晰图像的对比图，具体而言就是得到模糊和清晰图像的特征向量 feature1 = hist(q) 和 feature2，
然后再绘制 feature1 和 feature2 的对比折线图
"""
def local_kurtosis_contrast(img_path, blur_img_path, bins=512, patch_size=11):
    img = cv2.imread(img_path, 0)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    hist_no_blur = local_kurtosis(img, patch_size, bins=bins, blur=False)

    blur_img = cv2.imread(blur_img_path, 0)
    blur_img = (blur_img - np.min(blur_img)) / (np.max(blur_img) - np.min(blur_img))
    hist_blur = local_kurtosis(blur_img, patch_size, bins=bins, blur=True)

    x = np.linspace(1, bins, bins)
    plt.plot(x, hist_blur, label='blur', color='red', marker='o', linestyle='-')
    plt.plot(x, hist_no_blur, label='no blur', color='blue', marker='o', linestyle='-')
    plt.legend()
    plt.show()


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
    local_kurtosis_contrast('../input/376.jpg', '../input/376_blur.jpg')
    local_kurtosis_contrast('../input/499.jpg', '../input/499_blur.jpg')
    local_kurtosis_contrast('../input/445.jpg', '../input/445_blur.jpg')
    local_kurtosis_contrast('../input/386.jpg', '../input/386_blur.jpg')