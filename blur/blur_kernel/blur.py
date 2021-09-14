import numpy as np
import cv2
import math


# 生成卷积核和锚点
# noinspection PyShadowingNames
def generate_blur_kernel(length, angle):
    """
    生成一个模糊核/模糊矩阵
    :param length: 模糊矩阵的对角线的长度
    :param angle: 运动模糊的角度，取值范围为 -180 <= angle <= 180
                  1) 0 <= angle <= 90 时，往右上角移动
                  2) 90 < angle <= 180 时，往左上角移动
                  3) -90 <= angle < 0 时，往右下角移动
                  4) -180 <= angle < -90 时，往左下角移动
    :return:返回生成的核矩阵以及锚点
    """

    half = length / 2
    # eps 是取非负的最小值，一般当计算的 IOU 为 0 时，需要使用 eps 来进行替换
    # 最小的可表示数字，例如 1.0 + eps！= 1.0
    EPS = np.finfo(float).eps
    # math.floor 取整数部分，即：
    # math.floor(11.46) = math.floor(11.68) = math.floor(11.5) = 11
    # math.floor(-11.46) = math.floor(-11.68) = math.floor(-11.5) = -12
    # alpha 大于 180 度时，需要将其转变为 180 度以内，比如 200 度会变成 20 度
    alpha = (angle - math.floor(angle / 180) * 180) / 180 * math.pi
    cos_alpha = math.cos(alpha)
    sin_alpha = math.sin(alpha)

    if cos_alpha < 0 or cos_alpha < EPS:
        xsign = -1
    elif angle == 90:
        xsign = 0
    else:
        xsign = 1

    psfwdt = 1
    # 模糊核大小， width 为 length * cos，height 为 length * sin
    # length 的长度也为模糊核的对角线的长度
    sx = int(math.fabs(length * cos_alpha + psfwdt * xsign - length * EPS))
    sy = int(math.fabs(length * sin_alpha + psfwdt))
    psf = np.zeros((sy, sx))

    # psf 是左上角的权值较大，越往右下角权值越小的核，这时运动像是从右下角到左上角移动
    for i in range(0, sy):
        for j in range(0, sx):
            psf[i][j] = i * math.fabs(cos_alpha) - j * sin_alpha
            rad = math.sqrt(i * i + j * j)

            if rad >= half and math.fabs(psf[i][j]) <= psfwdt:
                temp = half - math.fabs((j + psf[i][j] * sin_alpha) / cos_alpha)
                psf[i][j] = math.sqrt(psf[i][j] * psf[i][j] + temp * temp)

            psf[i][j] = psfwdt + EPS - math.fabs(psf[i][j])

            if psf[i][j] > 0 and psf[i][j] > EPS:
                psf[i][j] += np.random.uniform(0, psf[i][j])

            if psf[i][j] < 0:
                psf[i][j] = 0

    # 运动方向默认是往左上运动，锚点在（0，0）
    anchor = (0, 0)
    # 运动方向是往右上角移动，锚点位于核矩阵的右上角，同时，左右翻转核矩阵，使得越靠近锚点，权值越大
    if 0 <= angle <= 90:
        # np.fliplr 把矩阵进行左右翻转
        psf = np.fliplr(psf)
        # anchor 位于核矩阵的右上角（width - 1, 0)
        anchor = (psf.shape[1] - 1, 0)
    # 运动方向是往右下角移动，锚点位于核矩阵的右下角，同时，上下以及左右翻转核矩阵，使得越靠近锚点，权值越大
    elif -90 <= angle < 0:
        psf = np.flipud(psf)
        psf = np.fliplr(psf)
        # anchor 位于核矩阵的右下角，(width - 1, height - 1)
        anchor = (psf.shape[1] - 1, psf.shape[0] - 1)
    # 运动方向是往左下角移动，锚点位于核矩阵的左下角，同时，上下翻转核矩阵，使得越靠近锚点，权值越大
    elif -180 <= angle < -90:
        psf = np.flipud(psf)
        # anchor 位于核矩阵的左下角，(0, height - 1)
        anchor = (0, psf.shape[0] - 1)

    psf = psf / psf.sum()

    return psf, anchor


if __name__ == '__main__':
    blur_length = 45

    kernel, anchor = generate_blur_kernel(blur_length, 35)
    img = cv2.imread('../feature/imgs/boat.jpeg')
    # 锚点 anchor 决定了卷积核相对于生成目标点的位置。遍历图像中的每一个像素，以每一个像素为锚点，
    # 按照相对位置生成卷积范围，和卷积核对应元素相乘再求和得到目标图像中对应像素的值
    motion_blur = cv2.filter2D(img, -1, kernel, anchor=anchor)
    cv2.imwrite('../feature/imgs/boat_35.jpeg', motion_blur)

    # kernel, anchor = generate_blur_kernel(blur_length, 90)
    # motion_blur = cv2.filter2D(img, -1, kernel, anchor=anchor)
    # cv2.imwrite('lenna_90.jpg', motion_blur)
    #
    # kernel, anchor = generate_blur_kernel(blur_length, 0)
    # motion_blur = cv2.filter2D(img, -1, kernel, anchor=anchor)
    # cv2.imwrite('lenna_0.jpg', motion_blur)
    # cv2.waitKey(0)
