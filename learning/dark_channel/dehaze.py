import cv2
import math
import numpy as np

def dark_channel(im, sz):
    # im 是彩色三通道图像的话，就将其拆分成三个通道，并且每一个通道都是灰度图像
    b, g, r = cv2.split(im)
    # 求出了 min(J_c(y)) c = r, g, b
    dc = cv2.min(cv2.min(r, g), b)

    # 使用腐蚀形态学操作来替代论文中的最小值滤波，最小值滤波等价于腐蚀，这是因为最小值滤波
    # 将一个窗口中的最小的像素（偏暗）赋值给窗口的中心值，扩大了较暗像素的范围，效果类似于腐蚀操作
    # when the minimum filter is applied to a digital image it picks up the minimum value of the neighbourhood pixel
    # window and assigns it to the current pixel. A pixel with the minimum value is the darkest among the pixels in
    # the pixel window.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz,sz))
    dark_ch = cv2.erode(dc, kernel)
    return dark_ch


# noinspection PyShadowingNames
def atm_light(img, dark):
    """
    我们可以通过借助于暗通道图来从有雾图像中获取该值。
    1.从暗通道图 dark 中按照亮度的大小取前 0.1% 的像素
    2.在这些位置中，在原始有雾图像 I 中寻找对应的具有最高亮度的点的值，作为 A 值
    :param img: 原始的有雾的图像
    :param dark: img 对应的暗通道图像
    :return:
    """
    h, w = img.shape[:2]
    img_size = h * w
    # img_size / 1000 表示前 0.1% 的像素点的个数
    numpx = int(max(math.floor(img_size / 1000), 1))
    darkvec = dark.reshape(img_size)
    imvec = img.reshape(img_size, 3)

    # 对暗通道中的像素值按从小到大进行排序，最后返回的是下标值
    indices = darkvec.argsort()
    # 返回暗通道中，亮度前 0.1% 的像素的下标值
    indices = indices[img_size - numpx::]
    atmsum = np.zeros([1, 3])

    # 在这些位置中，将原始有雾图像 I 中对应的像素点的值累加起来，然后求其平均值
    for i in range(1, numpx):
        atmsum = atmsum + imvec[indices[i]]

    A = atmsum / numpx

    return A

# 初步的估算透射率 t(x)
def transmission_estimate(img, A, sz):
    omega = 0.95
    img_tmp = np.empty(img.shape, img.dtype)

    # 计算 I_c(y) / A_c
    for i in range(0, 3):
        img_tmp[:, :, i] = img[:, :, i] / A[0, i]
    # 重新使用类似于暗通道的方法来进行计算
    transmission = 1 - omega * dark_channel(img_tmp, sz)
    # 返回的透射率图 t(x) 会和原始有雾图像（I）的使用导向滤波生成一个新的更精准的透射率图
    return transmission

def guided_filter(I, p, r, eps):

    mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * I + mean_b

    return q

def transmission_refine(im, et):
    # 将原始的有雾图像转变为灰度图像
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    r = 60
    eps = 0.0001
    # 使用导向滤波，生成一个更精确的透视图
    t = guided_filter(gray, et, r, eps)

    return t

# 根据求出的 A, t(x), 以及原始有雾图像 I(x)，进行恢复
def recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype)
    t = cv2.max(t, tx)

    for i in range(0, 3):
        res[:, :, i] = (im[:, :, i] - A[0, i]) / t + A[0, i]

    return res

if __name__ == '__main__':

    fn = 'jungle.png'
    src = cv2.imread(fn)

    I = src.astype('float64') / 255
 
    dark = dark_channel(I, 15)
    A = atm_light(I, dark)
    te = transmission_estimate(I, A, 15)
    t = transmission_refine(src, te)
    J = recover(I, t, A, 0.1)

    cv2.imshow("dark",dark)
    cv2.imshow("t",t)
    cv2.imshow('I',src)
    cv2.imshow('J',J)
    cv2.imwrite("./image/J.png",J*255)
    cv2.waitKey()
