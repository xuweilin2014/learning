import numpy as np

from cv.smooth import box_filter
from cv.image import to_32F


class GuidedFilter:
    """
    This is a factory class which builds guided filter according to the channel number of guided Input.
    The guided input could be gray image, color image, or multi-dimensional feature map.

    References: K.He, J.Sun, and X.Tang. Guided Image Filtering. TPAMI'12.
    """
    def __init__(self, I, radius, eps):
        """
        根据 I 的通道数，来决定是构造灰度图像的导向滤波还是多通道的导向滤波
        :param I: guided image or guided feature map
        :param radius: radius of filter
        :param eps: value controlling sharpness
        """

        if len(I.shape) == 2:
            self._filter = GrayGuidedFilter(I, radius, eps)
        else:
            self._filter = MultiDimGuidedFilter(I, radius, eps)

    def filter(self, p):
        """
        对输入的图像使用导向滤波进行处理，并且返回处理的结果
        :param p: filtering input which is 2D or 3D with format (height, width) or (height, width, channels)
        :return: filtering output whose shape is same with input
        """

        # 对文献中的代码 K.He 是通过 matlab 编写的，matlab 中读取图像时会自动将图像归一化为 0-1，
        # 因此正则项系数的设置也对应小很多，比如 0.1, 0.01 等。所以这里 to_32F 会对图像 p 进行归一化，除以 255
        p = to_32F(p)
        if len(p.shape) == 2:
            return self._filter.filter(p)
        elif len(p.shape) == 3:
            channels = p.shape[2]
            ret = np.zeros_like(p, dtype=np.float32)
            for c in range(channels):
                ret[:, :, c] = self._filter.filter(p[:, :, c])
            return ret

class GrayGuidedFilter:
    """
    specific guided filter for gray guided image.
    """
    def __init__(self, I, radius, eps):
        """
        对灰度图像的导向滤波器进行初始化
        :param I: 2D guided image
        :param radius: radius of filter
        :param eps: Value controlling sharpness
        """

        self.I = to_32F(I)
        self.radius = radius
        # 岭回归中正则项的系数
        self.eps = eps

    def filter(self, p):
        """
        导向滤波的真正步骤，p 在这里是一个二维的矩阵
        :param p: filtering input of 2 dimensional image
        :return: filtering output of 2 dimensional image
        """

        # step 1
        meanI = box_filter(I=self.I, r=self.radius)
        meanp = box_filter(I=p, r=self.radius)
        corrI = box_filter(I=self.I * self.I, r=self.radius)
        corrIp = box_filter(I=self.I * p, r=self.radius)
        # step 2
        varI = corrI - meanI * meanI
        covIp = corrIp - meanI * meanp
        # step 3
        a = covIp / (varI + self.eps)
        b = meanp - a * meanI
        # step 4
        meana = box_filter(I=a, r=self.radius)
        meanb = box_filter(I=b, r=self.radius)
        # step 5
        q = meana * self.I + meanb

        return q


class MultiDimGuidedFilter:
    """
    Specific guided filter for color guided image or multi-dimensional feature map.
    """
    def __init__(self, I, radius, eps):
        self.I = to_32F(I)
        self.radius = radius
        self.eps = eps

        self.rows = self.I.shape[0]
        self.cols = self.I.shape[1]
        self.chs = self.I.shape[2]

    def filter(self, p):
        """
        对三通道或者多通道的图像进行滤波，也就是彩色或者多通道图像
        :param p: filtering input of 2 dimensional image
        :return: filtering output of 2 dimensional image
        """

        # 这里 self.I 的 shape 是 (h, w, 3)，注意 p 在这里的 shape 是 (h_p, w_p)，也就是二维的
        I = self.I
        fullI = self.I

        # I_r、I_g、I_b 都是 (h, w)
        I_r, I_g, I_b = I[:, :, 0], I[:, :, 1], I[:, :, 2]
        h, w = p.shape[:2]
        N = box_filter(np.ones((h, w)), self.radius)

        mean_I_r = box_filter(I_r, self.radius) / N
        mean_I_g = box_filter(I_g, self.radius) / N
        mean_I_b = box_filter(I_b, self.radius) / N
        mean_p = box_filter(p, self.radius) / N

        # mean of I * p
        mean_Ip_r = box_filter(I_r * p, self.radius) / N
        mean_Ip_g = box_filter(I_g * p, self.radius) / N
        mean_Ip_b = box_filter(I_b * p, self.radius) / N

        # per-patch covariance of (I, p)
        covIp_r = mean_Ip_r - mean_I_r * mean_p
        covIp_g = mean_Ip_g - mean_I_g * mean_p
        covIp_b = mean_Ip_b - mean_I_b * mean_p

        # symmetric covariance matrix of I in each patch:
        # rr rg rb
        # rg gg gb
        # rb gb bb
        var_I_rr = box_filter(I_r * I_r, self.radius) / N - mean_I_r * mean_I_r
        var_I_rg = box_filter(I_r * I_g, self.radius) / N - mean_I_r * mean_I_g
        var_I_rb = box_filter(I_r * I_b, self.radius) / N - mean_I_r * mean_I_b
        var_I_gg = box_filter(I_g * I_g, self.radius) / N - mean_I_g * mean_I_g
        var_I_gb = box_filter(I_g * I_b, self.radius) / N - mean_I_g * mean_I_b
        var_I_bb = box_filter(I_b * I_b, self.radius) / N - mean_I_b * mean_I_b

        a = np.zeros((h, w, 3))
        for i in range(h):
            for j in range(w):
                sig = np.array([
                    [var_I_rr[i, j], var_I_rg[i, j], var_I_rb[i, j]],
                    [var_I_rg[i, j], var_I_gg[i, j], var_I_gb[i, j]],
                    [var_I_rb[i, j], var_I_gb[i, j], var_I_bb[i, j]]
                ])
                covIp = np.array([covIp_r[i, j], covIp_g[i, j], covIp_b[i, j]])
                a[i, j, :] = np.linalg.solve(sig + self.eps * np.eye(3), covIp)

        b = mean_p - a[:, :, 0] * mean_I_r - a[:, :, 1] * mean_I_g - a[:, :, 2] * mean_I_b

        meanA = box_filter(a, self.radius) / N[..., np.newaxis]
        meanB = box_filter(b, self.radius) / N

        # a 的 shape 为 (h,w,3)，所以需要把第 2 维的轴方向上的数值相加，最后变为 (h,w)
        q = np.sum(meanA * fullI, axis=2) + meanB

        return q
