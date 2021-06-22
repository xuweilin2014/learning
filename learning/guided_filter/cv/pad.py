import numpy as np


def padding_constant(image, pad_size, constant_value=0):
    """
    使用常数（默认为 0）在图像的边界进行填充：000000|abcdefg|000000
    :param image: image to padding. Only support 2D(gray) or 3D(color)
    :param pad_size: padding size for height and width axis respectively
    :param constant_value: 在图像边界处要填充的数值
    :return: image after padding
    """

    shape = image.shape
    assert len(shape) in [2, 3], 'image must be 2D or 3D'

    is_3D = True
    # 如果 image 中 shape 的长度为 2，那么就表明需要将其长度扩展为 3，增加一维
    if len(shape) == 2:
        image = np.expand_dims(image, axis=2)
        shape = image.shape
        is_3D = False

    # pad_size 为在 height 和 width 这两个方向分别需要进行填充的大小，在这里一般都为 r
    # 也就是 box filter 中窗口的半径
    h, w = pad_size
    # 将 image 由原来的 (h,w) 扩展为 (h + 2r, w + 2r)
    ret = np.zeros((shape[0] + 2 * h, shape[1] + 2 * w, shape[2]))

    # 将扩展的部分全部赋值为常数，而将原来的部分赋值为 image
    ret[h:-h, w:-w, :] = image
    ret[:h, :, :] = constant_value
    ret[-h:, :, :] = constant_value
    ret[:, :w, :] = constant_value
    ret[:, -w:, :] = constant_value

    return ret if is_3D else np.squeeze(ret, axis=2)


def padding_reflect(image, pad_size):
    """
    使用图像边界处的像素值来对扩展之后的图像进行填充:fedcba|abcdefg|gfedcb
    :param image: image to padding. Only support 2D(gray) or 3D(color)
    :param pad_size: padding size for height and width axis respectively
    :return: image after padding
    """
    pass


def padding_reflect_101(image, pad_size):
    """
    使用图像边界处的像素值来对扩展之后的图像进行填充:gfedcb|abcdefg|fedcba
    :param image: image to padding. Only support 2D(gray) or 3D(color)
    :param pad_size: padding size for height and width axis respectively
    :return: image after padding
    """
    pass


def padding_edge(image, pad_size):
    """
    使用图像边界处的像素值来对扩展之后的图像进行填充:aaaaaa|abcdefg|gggggg
    :param image: image to padding. Only support 2D(gray) or 3D(color)
    :param pad_size: padding size for height and width axis respectively
    :return: image after padding
    """

    shape = image.shape
    assert len(shape) in [2, 3], 'image must be 2D or 3D'

    is_3D = True
    # 如果 image 中 shape 的长度为 2，那么就表明需要将其长度扩展为 3，增加一维
    if len(shape) == 2:
        image = np.expand_dims(image, axis=2)
        shape = image.shape
        is_3D = False

    # pad_size 为在 height 和 width 这两个方向分别需要进行填充的大小，在这里一般都为 r
    # 也就是 box filter 中窗口的半径
    h, w = pad_size
    # 将 image 由原来的 (h,w) 扩展为 (h + 2r, w + 2r)
    ret = np.zeros((shape[0] + 2 * h, shape[1] + 2 * w, shape[2]))

    # 对于扩展之后的 (h + 2r, w + 2r) 图像，原图像的部分，就赋值为图像 image，而扩展的部分，则赋值为 image 边缘的像素点的值
    for i in range(shape[0] + 2 * h):
        for j in range(shape[1] + 2 * w):
            if i < h:
                if j < w:
                    ret[i, j, :] = image[0, 0, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[0, j - w, :]
                else:
                    ret[i, j, :] = image[0, shape[1] - 1, :]
            elif h <= i <= h + shape[0] - 1:
                if j < w:
                    ret[i, j, :] = image[i - h, 0, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[i - h, j - w, :]
                else:
                    ret[i, j, :] = image[i - h, shape[1] - 1, :]
            else:
                if j < w:
                    ret[i, j, :] = image[shape[0] - 1, 0, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[shape[0] - 1, j-w, :]
                else:
                    ret[i, j, :] = image[shape[0] - 1, shape[1] - 1, :]

    return ret if is_3D else np.squeeze(ret, axis=2)

