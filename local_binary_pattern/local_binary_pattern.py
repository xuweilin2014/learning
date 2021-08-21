import sys
import numpy as np
import cv2 as cv
from numba import jit

image_path = "cat.png"

# 在圆形选取框基础上，加入旋转不变操作
@jit(cache=True)
def rotation_invariant_LBP(img, radius=3, neighbors=8):
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
                w1 = (1-tx) * (1-ty)
                w2 = tx * (1-ty)
                w3 = (1-tx) * ty
                w4 = tx * ty
                # 根据双线性插值公式计算第 k 个采样点的灰度值
                neighbor = img[i + y1, j + x1] * w1 + img[i + y2,j + x1] * w2 + img[i + y1,j + x2] * w3 + img[i + y2, j + x2] * w4
                # LBP 特征图像的每个邻域的 LBP 值累加，累加通过与操作完成，对应的 LBP 值通过移位取得
                dst[i - radius,j - radius] |= (neighbor > center) << np.uint8(neighbors - k - 1)

    # 进行旋转不变处理
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            currentValue = dst[i,j]
            minValue = currentValue
            for k in range(1, neighbors):
                # 对二进制编码进行循环左移，意思即选取移动过程中二进制码最小的那个作为最终值
                temp = np.uint8(currentValue >> (neighbors - k)) | np.uint8(currentValue << k)
                if temp < minValue:
                    minValue = temp
            dst[i,j] = minValue

    return dst


if __name__ == '__main__':
    gray = cv.imread('peppers.jpg', cv.IMREAD_GRAYSCALE)
    rotation_invariant = rotation_invariant_LBP(gray, 3, 8)
    cv.imshow('img', gray)
    cv.imshow('ri', rotation_invariant)
    cv.waitKey(0)
    cv.destroyAllWindows()