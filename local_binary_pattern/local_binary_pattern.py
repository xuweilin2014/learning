import cv2 as cv
import numpy as np
from numba import jit
import matplotlib.pyplot as plt

image_path = "cat.png"

# 在圆形选取框基础上，加入旋转不变操作，生成局部二进制图像，并且计算 LBP 图像的特征向量或者说直方图特征
@jit(cache=True)
def rotation_invariant_LBP(img, radius=3, neighbors=8, span=15):
    h, w = img.shape
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
                neighbor = img[i + y1, j + x1] * w1 + img[i + y1,j + x2] * w2 + img[i + y2,j + x1] * w3 + img[i + y2, j + x2] * w4
                # LBP 特征图像的每个邻域的 LBP 值累加，累加通过与操作完成，对应的 LBP 值通过移位取得
                dst[i - radius, j - radius] |= (neighbor > center) << np.uint8(neighbors - k - 1)

    '''
    对于 8 采样点的 LBP 算子来说，特征值范围为 0~255，对每个特征值进行统计，比如得到特征值为 1 的 LBP 值有多少个、
    特征值为 245 的 LBP 值有多少个等等。这样就形成了一个直方图，该直方图有 256 个 bin，即 256 个分量，也可以把该直方图
    当做一个长度为 256 的向量。
    如果直接使用该向量的话，那么对 8 采样点的 LBP 算子来说，一张图片至多会形成一个 256 长度的一个向量，这样位置信息就全部丢失了，
    会造成很大的精度问题。所以在实际中还会再有一个技巧，就是先把图像分成若干个区域，对每个区域进行统计得到直方图向量，再将这些向
    量整合起来形成一个大的向量。
    在这里就是将图像分成不同的小区域，每一个区域有 (span*span) 个像素，每一个区域生成一个含有 256 个分量的直方图或者说特征向量，然后将
    所有的区域的特征向量合并起来并且正则化，最后得到图像的特征向量
    '''

    length = 1 << neighbors
    rows = int(np.ceil(h / span))
    cols = int(np.ceil(w / span))
    hist = np.zeros(length * rows * cols)

    # 进行旋转不变处理
    # Maenpaa 等人又将 LBP算子进行了扩展，提出了具有旋转不变性的 LBP 算子，
    # 即不断旋转圆形邻域得到一系列初始定义的 LBP值，取其最小值作为该邻域的 LBP 值。
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            cur_val = dst[i, j]
            min_val = cur_val
            for k in range(1, neighbors):
                # 对二进制编码进行【循环左移】，意思即选取移动过程中二进制码最小的那个作为最终值
                temp = np.uint8(cur_val >> (neighbors - k)) | np.uint8(cur_val << k)
                if temp < min_val:
                    min_val = temp

            dst[i, j] = min_val
            r = int(np.floor(i / span))
            c = int(np.floor(j / span))
            index = int(min_val / 255 * length)
            index = index - 1 if index == length else index
            hist[(r * rows + c) * length + index] += 1

    plt.title('LBP histogram')
    plt.plot(hist)
    plt.show()

    return dst

if __name__ == '__main__':
    gray = cv.imread('peppers.jpg', cv.IMREAD_GRAYSCALE)
    rotation_invariant = rotation_invariant_LBP(gray, 3, 8)
    cv.imshow('img', gray)
    cv.imshow('ri', rotation_invariant)
    cv.waitKey(0)
    cv.destroyAllWindows()