import numpy as np

from pad import *


def box_filter(I, r, normalize=True, border_type='edge'):
    """
    box filter 平滑滤波，采用积分图算法实现
    :param I: input should be 3D with format of (height, width, channels)
    :param r: radius of the filter. kernel size = 2 * r + 1
    :param normalize: whether to normalize
    :param border_type: border type for padding, includes:
                        edge        :   aaaaaa|abcdefg|gggggg
                        zero        :   000000|abcdefg|000000
                        reflect     :   fedcba|abcdefg|gfedcb
                        reflect_101 :   gfedcb|abcdefg|fedcba
    :return: output has same shape with input
    """

    I = I.astype(np.float32)
    shape = I.shape
    assert len(shape) in [2, 3], "I should be NDArray of 2D or 3D, not %dD" % len(shape)
    is_3D = True

    # 如果 image 中 shape 的长度为 2，那么就表明需要将其长度扩展为 3，增加一维
    if len(shape) == 2:
        I = np.expand_dims(I, axis=2)
        shape = I.shape
        is_3D = False

    (rows, cols, channels) = shape
    tmp = np.zeros(shape=(rows, cols + 2 * r, channels), dtype=np.float32)
    ret = np.zeros(shape=shape, dtype=np.float32)

    # 根据边界填充的类型，来对 I 进行边界填充，默认使用 zero 边界填充的方式
    # 即先对 I 的 shape 进行扩展变大，然后再对边界上的像素值进行填充，经过 padding 之后，I 的 shape 变为 (h + 2r, w + 2r)
    if border_type == 'reflect_101':
        I = padding_reflect_101(I, pad_size=(r, r))
    elif border_type == 'reflect':
        I = padding_reflect(I, pad_size=(r, r))
    elif border_type == 'edge':
        I = padding_edge(I, pad_size=(r, r))
    elif border_type == 'zero':
        I = padding_constant(I, pad_size=(r, r), constant_value=0)
    else:
        raise NotImplementedError

    # (rows + 2r, cols + 2r)
    # axis = 0，表示按照列进行累加，比如 [[1,2,3],[4,5,6]]，经过按行累加之后，会得到 [[1,2,3],[5,7,9]]
    I_cum = np.cumsum(I, axis=0)
    tmp[0, :, :] = I_cum[2 * r, :, :]
    # tmp[i, :, :] = I_cum[2 * r + i, :, :] - I_cum[i - 1, :, :]
    # 此时 tmp[i, j, :] 表示的是 I[i - r][j] 到 I[i + r][j] 之间的像素值之和
    tmp[1:rows, :, :] = I_cum[2 * r + 1:2 * r + rows, :, :] - I_cum[0:rows - 1, :, :]

    # axis = 1，表示按照行进行累加，比如 [[1,2,3],[4,5,6]]，经过按行累加之后，会得到 [[1,3,6],[4,9,15]]
    I_cum = np.cumsum(tmp, axis=1)
    ret[:, 0, :] = I_cum[:, 2 * r, :]
    # tmp[:, j, :] = I_cum[:, 2 * r + j, :] - I_cum[:, j - 1, :]
    # 此时，ret[i, j, :] 表示的是 tmp[i][j - r] 到 tmp[i][j + r] 之间的和，由于 tmp[i][j] 表示的为 tmp[i - r,j] 到 tmp[i + r,j] 之间的和，
    # 那么 ret[i][j] 就表示 I 中以点 (i,j) 为中心，以 r 为半径的窗口中像素值之和
    ret[:, 1:cols, :] = I_cum[:, 2 * r + 1:2 * r + cols, :] - I_cum[:, 0:cols - 1, :]

    if normalize:
        ret /= float((2 * r + 1) ** 2)

    return ret if is_3D else np.squeeze(ret, axis=2)


def blur(I, r):
    """
    使用 box filter 进行平滑滤波
    :param I: filtering input
    :param r: radius of blur filter
    :return: blurred output of I
    """
    ones = np.ones_like(I, dtype=np.float32)
    # 求出的二维矩阵 N 中，每一个点 (i,j) 的值表示以点为中心，以 r 为半径的窗口中，包含的像素点的个数
    N = box_filter(ones, r)
    # box filter 对 I 进行求和
    ret = box_filter(I, r)

    # ret / N 表示进行平滑
    return ret / N
