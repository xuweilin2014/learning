# -*- coding: utf-8 -*-
import cv2
import numpy as np

def harris_detect(img, ksize=3):
    """
    自己实现角点检测

    params:
        img:灰度图片
        ksize：Sobel算子窗口大小

    return：
        corner：与源图像一样大小，角点处像素值设置为255
    """
    k = 0.04  # 响应函数 R 中的系数 k
    threshold = 0.01  # 设定阈值
    WITH_NMS = False  # 是否非极大值抑制

    # 1、使用 Sobel 计算像素点 x,y 方向的梯度
    h, w = img.shape[:2]
    # Sobel 函数求完导数后会有负值，还有会大于 255 的值。而原图像是 uint8，即 8 位无符号数
    # 所以 Sobel 建立的图像位数不够，会有截断。因此要使用 16 位有符号的数据类型，即 cv2.CV_16S
    # cv2.Sobel(src, ddepth, dx, dy[, ksize])
    # src: 输入图像
    # ddepth: 输出图像深度
    # dx, dy: dx = 1, dy = 0 时求 x 方向的一阶导数，当组合为 dx = 0, dy = 1 时，求 y 方向的一阶导数
    # ksize: Sobel 算子的大小，必须是 1,3,5 或者 7
    grad = np.zeros((h, w, 2), dtype=np.float32)
    # 计算 Ix
    grad[:, :, 0] = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
    # 计算 Iy
    grad[:, :, 1] = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)

    # 2、计算 Ix ^ 2, Iy ^ 2, Ix * Iy
    m = np.zeros((h, w, 3), dtype=np.float32)
    m[:, :, 0] = grad[:, :, 0] ** 2
    m[:, :, 1] = grad[:, :, 1] ** 2
    m[:, :, 2] = grad[:, :, 0] * grad[:, :, 1]

    # 3、利用高斯函数对 Ix ^ 2, Iy ^ 2, Ix * Iy 进行滤波
    m[:, :, 0] = cv2.GaussianBlur(m[:, :, 0], ksize=(ksize, ksize), sigmaX=2)
    m[:, :, 1] = cv2.GaussianBlur(m[:, :, 1], ksize=(ksize, ksize), sigmaX=2)
    m[:, :, 2] = cv2.GaussianBlur(m[:, :, 2], ksize=(ksize, ksize), sigmaX=2)
    m = [np.array([[m[i, j, 0], m[i, j, 2]], [m[i, j, 2], m[i, j, 1]]]) for i in range(h) for j in range(w)]

    # 4、计算局部特征结果矩阵 M 的特征值和响应函数 R(i,j) = det(M) - k(trace(M))^2  0.04 <= k <= 0.06
    D, T = list(map(np.linalg.det, m)), list(map(np.trace, m))
    R = np.array([d - k * t ** 2 for d, t in zip(D, T)])

    # 5、将计算出响应函数的值 R 要满足大于设定的阈值, 获取最大的 R 值
    R_max = np.max(R)
    R = R.reshape(h, w)
    corner = np.zeros_like(R, dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            # 只进行阈值检测
            if R[i, j] > R_max * threshold:
                corner[i, j] = 255

    return corner


if __name__ == '__main__':
    img = cv2.imread('./example.jpg')
    img = cv2.resize(img, dsize=(600, 400))
    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = harris_detect(gray)
    print(dst.shape)  # (400, 600)
    img[dst > 0.01 * dst.max()] = [0, 0, 255]
    cv2.imshow('', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()