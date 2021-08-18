import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
from skimage.transform import radon
from sklearn.metrics import mean_squared_error
from skimage.filters import window
from statsmodels.graphics import tsaplots
import librosa, librosa.display


# 将图像转换到频域
def convert_frequency(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    # 取绝对值：将复数变化成实数
    # 取对数，目的为了将数据变化到较小的范围（比如 0-255）
    abs1 = np.abs(f)
    log1 = np.log(abs1)

    abs2 = np.abs(fshift)
    log2 = np.log(abs2)

    plt.subplot(231),plt.imshow(img, 'gray'),plt.title('original')
    plt.subplot(232), plt.imshow(abs1, 'gray'), plt.title('fft2')
    plt.subplot(233), plt.imshow(abs2, 'gray'), plt.title('shift2center')
    plt.subplot(235), plt.imshow(log1, 'gray'), plt.title('log_fft2')
    plt.subplot(236), plt.imshow(log2, 'gray'), plt.title('log_fft2center')

    plt.show()

# 将图像 img 投影到直线 l 上，这条直线 l 与 x 轴的夹角为 angle (0 <= angle <= 90)，并且经过原点
# 设与直线 l 垂直，并且垂点也位于原点的直线为 z，直线 z 将图像矩阵分为两部分，左侧和右侧，左侧和右侧
# 分别采用不同的方式进行处理，判断标准就是 x / y 和 tan(theta) 的大小
def project_line(img, angle):
    height, width = img.shape
    theta = (angle / 180) * math.pi
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    tan_theta = math.tan(theta)

    # length 表示的是图像矩阵在直线 l 上投影的最大间距
    length = int(height * sin_theta + width * cos_theta)
    xl = np.linspace(1, length, length, dtype=int)
    yl = np.empty(length)

    # 遍历图像矩阵，图像 height 看做是纵轴 y，图像 width 看做是横轴 x
    for y in range(1, height + 1):
        for x in range(1, width + 1):
            d = np.sqrt(y * y + x * x)
            base = height * sin_theta

            # 当 x / y < tan(theta)，那么说明图像中的点位于直线 z 的左侧
            if x / y < tan_theta:
                index = base - d * math.cos(math.atan(x / y) + math.pi / 2 - theta)
            # 当 x / y > tan(theta)，那么说明图像中的点位于直线 z 的右侧
            elif x / y > tan_theta:
                index = d * math.cos(math.atan(y / x) + theta) + base
            # 当 x / y = tan(theta)，那么说明图像中的点位于直线 z 上
            else:
                index = base
            index = int(index)
            if index == length:
                index -= 1
            elif index < 0:
                index = 0

            yl[math.floor(index)] += img[y - 1, x - 1]
    
    return xl, yl

# 将图像 image 进行 radon 变换
def discrete_radon_transform(image):
    image = zoom(image, 0.4)

    plt.figure(figsize=(10, 5))

    plt.subplot(221)
    plt.title("Original")
    plt.imshow(image, cmap=plt.cm.Greys_r)

    plt.subplot(222)
    # 将图像投影到通过原点，并且与 x 轴的角度为 theta 度的直线上
    projection = radon(image, theta=[60])
    plt.plot(projection)
    plt.title("Projections at 60 degree")
    plt.xlabel("Projection axis")
    plt.ylabel("Intensity")

    projections = radon(image)
    plt.subplot(223)
    plt.title("Radon transform\n(Sinogram)")
    plt.xlabel("Projection axis")
    plt.ylabel("Intensity")
    plt.imshow(projections)

    plt.subplots_adjust(hspace=0.4, wspace=0.5)
    plt.show()

# 使用三阶函数来对 radon 变换的结果进行拟合
# 在运动模糊的方向，拟合结果与 radon 变换结果的 MSE 误差最大
def poly_fit(image):
    image = zoom(image, 0.4)

    # 对图像 image 进行 radon 变换，得到 [distance, 180]
    # 其中 distance 就是投影到某一平面的数据分布长度，而 180 就是指沿 180 个角度进行投影
    projections = radon(image)
    xlen, theta_len = projections.shape
    xlen = xlen + 1 if xlen % 2 != 0 else xlen
    xlen = int(xlen / 2)

    dis = int(min(image.shape) / (np.sqrt(2) * 2))

    x = np.linspace(1, dis, dis, dtype=int)
    errors = np.empty(theta_len)
    for i in range(theta_len):
        projection = projections[xlen:xlen + dis, i]
        # 使用 3 阶多项式进行数据拟合，得到了多项式的系数
        z = np.polyfit(x, projection, 3)
        p = np.poly1d(z)
        # 使用 3 阶多项式系数生成拟合后的 y 值
        y_vals = p(x)
        # 计算 MSE 误差
        errors[i] = mean_squared_error(projection, y_vals)

    thetas = np.linspace(0, theta_len, theta_len)
    plt.plot(thetas, errors)
    plt.title('residual figure')
    plt.show()

if __name__ == '__main__':
    # 直接读为灰度图像
    img = cv2.imread('imgs/cameraman_60.jpg', 0)
    # 在时域对图像加上汉宁窗，否则在计算 fft 过程中，由于周期延拓造成图像边缘像素的不连续，
    # 在图像的频谱图中会产生亮十字线，影响到对运动角度的检测
    img = img * window('hann', img.shape)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    # 取绝对值：将复数变化成实数
    # 取对数：目的为了将数据变化到较小的范围（比如 0-255）
    img_f = np.log(np.abs(fshift))
    img_f = img_f.astype(np.uint8)
    img_f = cv2.medianBlur(img_f, 3)
    discrete_radon_transform(img_f)