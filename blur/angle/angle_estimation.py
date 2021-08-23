import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
from skimage.transform import radon
from sklearn.metrics import mean_squared_error
from skimage.filters import window
from blur_kernel.blur import generate_blur_kernel
import numba

EPSILON = 1.19209e-09
BINS = 30

class ColorHistogram:
    One = 1
    Two = 2

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
def poly_fit_angle(image):
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

    # thetas = np.linspace(0, theta_len, theta_len)
    # plt.plot(thetas, errors)
    # plt.title('residual figure')
    # plt.show()
    return np.argmax(errors)

# noinspection PyShadowingNames
@numba.jit(cache=True)
def mean_shift_length(img, angle, original_img, radius=3, neighbors=8):
    if len(img.shape) == 2:
        img = np.reshape(img, img.shape + (1,))

    h, w, _ = img.shape
    length = np.int(np.sqrt(np.power(h, 2) + np.power(w, 2)) / 2)
    # 线性搜索的步长默认为 4
    step = 7

    # 计算目标模型的颜色直方图特征
    target_color = color_distribution(img, ColorHistogram.One)
    # 计算目标模型的 LBP 直方图特征
    target_lbp = local_binary_pattern(img, radius, neighbors)
    target_model = np.concatenate((target_color, target_lbp), axis=0)
    max_coef = 0
    blur_len = 1

    # 模糊的长度就可以通过线性搜索的方式来获取, 步长为 step
    for k in range(1, length + 1, step):
        # 依据不同的模糊长度 len 产生不同的模糊核 kernel
        kernel, anchor = generate_blur_kernel(k, angle)
        motion_blur = cv2.filter2D(original_img, -1, kernel, anchor=anchor)
        candidate_color = color_distribution(motion_blur, ColorHistogram.One)
        candidate_lbp = local_binary_pattern(motion_blur, radius, neighbors)
        candidate_model = np.concatenate((candidate_color, candidate_lbp), axis=0)
        # 计算当前模糊长度生成的图片的颜色直方图模型与目标模型的相似度
        coef = compute_bhattacharyya_coef(target_model, candidate_model)
        if coef > max_coef and k != 1:
            blur_len = k
            max_coef = coef

    return blur_len

def compute_bhattacharyya_coef(p, q):
    return np.sum(np.sqrt(p * q))

@numba.jit(cache=True)
def color_distribution(img_patch, DIMENSION):
    bins = BINS
    h, w, c = img_patch.shape

    if DIMENSION == ColorHistogram.One:
        # 如果使用一维的颜色直方图特征
        # cd 即为颜色直方图特征向量，如果 c = 3，那么 cd 为 bins * 3，也就是将 rgb 颜色向量直接拼接
        cd = np.zeros(bins * c)
    else:
        # 如果使用二维的颜色直方图特制的
        # cd 应该为一个 (bins, bins) 大小的矩阵，并且第一维是 hue 特征，第二维是 saturation 特征
        cd = np.zeros((bins, bins))
        img_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2HSV)

    center = np.round(np.array([w / 2, h / 2]))
    dist = np.zeros((h, w))

    # compute distances
    # 计算 patch 中各点到 patch 中心点之间的距离
    xv, yv = np.linspace(1, w, w), np.linspace(1, h, h)
    x, y = np.meshgrid(xv, yv)
    dist = np.sqrt(np.power(x - center[0], 2) + np.power(y - center[1], 2))

    # normalize the distances
    dist = dist / np.max(dist)

    # 下面把 1-dimension 和 2-dimension 区分开是为了方便 @jit 进行加速处理
    # build the histogram and weight with the kernel
    if DIMENSION == ColorHistogram.One:
        for i in range(h):
            for j in range(w):
                # use Epachnikov kernel function
                # 如果 ||x|| <= 1, k(x) = 1 - x
                # 如果 ||x|| > 1, k(x) = 0
                if dist[i, j] ** 2 < 1:
                    kE = 2 / np.pi * (1 - dist[i, j] ** 2)
                else:
                    kE = 0

                for k in range(c):
                    # compute which bin is corresponding to the pixel value
                    index = np.int(img_patch[i, j, k] / 255 * bins)
                    index = index - 1 if index == bins else index

                    # add the kernel value to the bin
                    cd[k * bins + index] += kE

    if DIMENSION == ColorHistogram.Two:
        for i in range(h):
            for j in range(w):
                if dist[i, j] ** 2 < 1:
                    kE = 2 / np.pi * (1 - dist[i, j] ** 2)
                else:
                    kE = 0
                # 如果使用二维的颜色直方图特征，也就是一个 hue-saturation 矩阵
                # img_patch[i,j,0] 表示 hsv 中的 hue 分量，img_patch[i,j,1] 表示 hsv 中的 saturation 分量
                h_index = np.int(img_patch[i, j, 0] / 180 * bins)
                s_index = np.int(img_patch[i, j, 1] / 255 * bins)
                h_index = h_index - 1 if h_index == bins else h_index
                s_index = s_index - 1 if s_index == bins else s_index

                cd[h_index, s_index] += kE

    # normalize the kernel value
    cd = cd / np.sum(cd)
    cd[cd <= EPSILON] = EPSILON

    return cd

@numba.jit(cache=True)
def local_binary_pattern(img, radius=3, neighbors=8, span=15):
    img = img[:,:,0]
    h, w = img.shape

    length = 1 << neighbors
    rows = int(np.ceil(h / span))
    cols = int(np.ceil(w / span))
    hist = np.zeros(length * rows * cols)

    # LBP 算法是根据半径为 radius 的邻域的圆中所有点，计算出一个值，
    # 所以新生成的图像高和宽相比于之前都会减少 radius
    dst = np.zeros((h - 2 * radius, w - 2 * radius), dtype=img.dtype)
    for i in range(radius, h - radius):
        for j in range(radius, w - radius):
            # 获得中心像素点的灰度值
            center = img[i, j]
            for k in range(neighbors):
                # 计算采样点对于中心点坐标的偏移量 rx，ry
                rx = radius * np.cos(2.0 * np.pi * k / neighbors)
                ry = radius * np.sin(2.0 * np.pi * k / neighbors)
                # 为双线性插值做准备
                # 对采样点偏移量分别进行上下取整
                x1 = int(np.floor(rx))
                x2 = int(np.ceil(rx))
                y1 = int(np.floor(ry))
                y2 = int(np.ceil(ry))
                # 将坐标偏移量映射到 0-1 之间
                tx = rx - x1
                ty = ry - y1
                # 根据 0-1 之间的 x，y 的权重计算公式计算权重，权重与坐标具体位置无关，与坐标间的差值有关
                w1 = (1 - tx) * (1 - ty)
                w2 = tx * (1 - ty)
                w3 = (1 - tx) * ty
                w4 = tx * ty
                # 根据双线性插值公式计算第 k 个采样点的灰度值
                neighbor = img[i + y1, j + x1] * w1 + img[i + y1, j + x2] * w2 + img[i + y2, j + x1] * w3 + img[i + y2, j + x2] * w4
                # LBP 特征图像的每个邻域的 LBP 值累加，累加通过与操作完成，对应的 LBP 值通过移位取得
                dst[i - radius, j - radius] |= (neighbor > center) << np.uint8(neighbors - k - 1)

            # 进行旋转不变处理
            # Maenpaa 等人又将 LBP算子进行了扩展，提出了具有旋转不变性的 LBP 算子，
            # 即不断旋转圆形邻域得到一系列初始定义的 LBP值，取其最小值作为该邻域的 LBP 值。
            cur_val = dst[i - radius, j - radius]
            min_val = cur_val
            for k in range(1, neighbors):
                # 对二进制编码进行【循环左移】，意思即选取移动过程中二进制码最小的那个作为最终值
                temp = np.uint8(cur_val >> (neighbors - k)) | np.uint8(cur_val << k)
                if temp < min_val:
                    min_val = temp

            # 在这里就是将图像分成不同的小区域，每一个区域有 (span*span) 个像素，每一个区域生成一个含有 256 个分量的直方图或者说特征向量，然后将
            # 所有的区域的特征向量合并起来并且正则化，最后得到图像的特征向量
            r = int(np.floor(i / span))
            c = int(np.floor(j / span))
            index = int(min_val / 255 * length)
            index = index - 1 if index == length else index
            hist[(r * rows + c) * length + index] += 1

    hist = hist / np.sum(hist)
    return hist

if __name__ == '__main__':
    # 直接读为灰度图像
    original_img = cv2.imread('imgs/peppers.jpg')
    img_bgr = cv2.imread('imgs/peppers_60.jpg')
    cv2.imshow('bgr', img_bgr)
    img = cv2.imread('imgs/peppers_60.jpg', 0)
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
    angle = poly_fit_angle(img_f)
    len = mean_shift_length(img_bgr, angle, original_img)
    print(angle, len)