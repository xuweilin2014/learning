import numpy as np
import cv2


# used for linear mapping...
# 进行归一化操作
def linear_mapping(img):
    return (img - img.min()) / (img.max() - img.min())


# pre-processing the image...
"""
该函数对数据进行预处理：
1.对数据矩阵取对数
2.接着标准化数据，使得其更加符合标准正态分布 (经过这两步处理，直观上来说数据变得中心化了，弱化了背景的影响)
3.使用窗函数处理数据，减弱其频谱泄漏的现象
"""


def pre_process(img):
    # get the size of the img...
    height, width = img.shape
    img = np.log(img + 1)
    img = (img - np.mean(img)) / (np.std(img) + 1e-5)
    # use the hanning window...
    window = window_func_2d(height, width)
    img = img * window

    return img


def window_func_2d(height, width):
    win_col = np.hanning(width)
    win_row = np.hanning(height)
    mask_col, mask_row = np.meshgrid(win_col, win_row)

    win = mask_col * mask_row

    return win


'''
该函数对第一帧的目标框进行随即重定位，刚性形变，减轻漂移
'''
def random_warp(img):
    a = -180 / 16
    b = 180 / 16
    r = a + (b - a) * np.random.uniform()
    # rotate the image...
    matrix_rot = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), r, 1)
    img_rot = cv2.warpAffine(np.uint8(img * 255), matrix_rot, (img.shape[1], img.shape[0]))
    img_rot = img_rot.astype(np.float32) / 255
    return img_rot
