import numpy as np
import cv2
import fhog


# 离散傅立叶、逆变换
def fftd(img, backwards=False):
    # shape of img can be (m,n), (m,n,1) or (m,n,2)
    # in my test, fft provided by numpy and scipy are slower than cv2.dft
    return cv2.dft(np.float32(img), flags=((cv2.DFT_INVERSE | cv2.DFT_SCALE) if backwards else cv2.DFT_COMPLEX_OUTPUT))   # 'flags =' is necessary!


# 实部图像
def real(img):
    return img[:, :, 0]


# 虚部图像
def imag(img):
    return img[:, :, 1]


# 两个复数，它们的积 (a + bi)(c + di) = (ac - bd) + (ad + bc)i
def complexMultiplication(a, b):
    res = np.zeros(a.shape, a.dtype)

    res[:, :, 0] = a[:, :, 0] * b[:, :, 0] - a[:, :, 1] * b[:, :, 1]
    res[:, :, 1] = a[:, :, 0] * b[:, :, 1] + a[:, :, 1] * b[:, :, 0]
    return res


# 两个复数，它们相除 (a + bi) / (c + di) = (ac + bd) / (c*c + d*d) + ((bc - ad) / (c*c + d*d)) * i
def complexDivision(a, b):
    res = np.zeros(a.shape, a.dtype)
    divisor = 1. / (b[:, :, 0]**2 + b[:, :, 1]**2)

    res[:, :, 0] = (a[:, :, 0] * b[:, :, 0] + a[:, :, 1] * b[:, :, 1]) * divisor
    res[:, :, 1] = (a[:, :, 1] * b[:, :, 0] + a[:, :, 0] * b[:, :, 1]) * divisor
    return res


# 可以将 fft 输出中的直流分量移动到频谱的中央
def rearrange(img):
    # 断言必须为真，否则会抛出异常，ndim 为数组维数
    assert(img.ndim == 2)
    img_ = np.zeros(img.shape, img.dtype)
    xh, yh = img.shape[1] // 2, img.shape[0] // 2
    img_[0:yh, 0:xh], img_[yh:img.shape[0], xh:img.shape[1]] = img[yh:img.shape[0], xh:img.shape[1]], img[0:yh, 0:xh]
    img_[0:yh, xh:img.shape[1]], img_[yh:img.shape[0], 0:xh] = img[yh:img.shape[0], 0:xh], img[0:yh, xh:img.shape[1]]
    return img_


# rect tools
def x2(rect):
    return rect[0] + rect[2]


def y2(rect):
    return rect[1] + rect[3]


# limit 的值一定为 [0, 0, image.width, image.height]
def limit(rect, limit):
    if rect[0] + rect[2] > limit[0] + limit[2]:
        rect[2] = limit[0] + limit[2] - rect[0]
    if rect[1] + rect[3] > limit[1] + limit[3]:
        rect[3] = limit[1] + limit[3] - rect[1]
    # 如果 rect[0] 也就是 x 是小于 0 的，说明 rect 图像有一部分是在 image 图像之外，那么就将 width 调整为在图像内的长度
    if rect[0] < limit[0]:
        rect[2] -= (limit[0] - rect[0])
        rect[0] = limit[0]
    # 如果 rect[1] 也就是 y 是小于 0 的，说明 rect 图像有一部分是在 image 图像之外，那么就将 height 调整为在图像内的长度
    if rect[1] < limit[1]:
        rect[3] -= (limit[1] - rect[1])
        rect[1] = limit[1]
    if rect[2] < 0:
        rect[2] = 0
    if rect[3] < 0:
        rect[3] = 0
    return rect


def getBorder(original, limited):
    res = [0, 0, 0, 0]
    res[0] = limited[0] - original[0]
    res[1] = limited[1] - original[1]
    res[2] = x2(original) - x2(limited)
    res[3] = y2(original) - y2(limited)
    assert(np.all(np.array(res) >= 0))
    return res


# 经常需要空域或频域的滤波处理，在进入真正的处理程序前，需要考虑图像边界情况。
# 通常的处理方法是为图像增加一定的边缘，以适应【卷积核】在原图像边界的操作。
def subwindow(img, window, borderType=cv2.BORDER_CONSTANT):
    cutWindow = [x for x in window]
    limit(cutWindow, [0, 0, img.shape[1], img.shape[0]])   # modify cutWindow
    assert(cutWindow[2] > 0 and cutWindow[3] > 0)
    border = getBorder(window, cutWindow)
    res = img[cutWindow[1]:cutWindow[1] + cutWindow[3], cutWindow[0]:cutWindow[0] + cutWindow[2]]
    extract_image(img, cutWindow)

    # 由于 roi 区域可能会超出原图像边界，因此超出边界的部分填充为原图像边界的像素
    if border != [0, 0, 0, 0]:
        # 在 OpenCV 的滤波算法中，copyMakeBorder 是一个非常重要的工具函数，它用来扩充 res 图像的边缘，将图像变大，然后以各种
        # 外插的方式自动填充图像边界，这个函数实际上调用了函数 cv2.borderInterpolate，这个函数最重要的功能就是为了处理边界
        # borderType 是扩充边缘的类型，就是外插的类型，这里使用的是 BORDER_REPLICATE，也就是复制法，也就是复制最边缘像素
        res = cv2.copyMakeBorder(res, border[1], border[3], border[0], border[2], borderType)
        cv2.imwrite('res.jpg', res)
    return res


# KCF tracker
# 计算一维亚像素的峰值
def subPixelPeak(left, center, right):
    divisor = 2 * center - right - left  # float
    return 0 if abs(divisor) < 1e-3 else 0.5 * (right - left) / divisor


class KCFTracker:
    def __init__(self, hog=False, fixed_window=True, multiscale=False):
        # 岭回归中的 lambda 常数，正则化
        self.lambdar = 0.0001   # regularization
        # extra area surrounding the target
        # 在目标框附近填充的区域
        self.padding = 2.5
        # bandwidth of gaussian target
        self.output_sigma_factor = 0.125   # bandwidth of gaussian target

        # 是否使用 fhog 特征
        if hog:
            # interp_factor 可以看成是 learning  rate，用于对核相关滤波的 alpha 参数以及 fhog 特征矩阵进行在线更新
            self.interp_factor = 0.012   # linear interpolation factor for adaptation、
            # gaussian kernel bandwidth
            # 高斯卷积核的标准差
            self.sigma = 0.6
            # hog 元胞数组尺寸
            # Hog cell size
            self.cell_size = 4
            self._hogfeatures = True
        # 否则使用图像灰度特征
        # raw gray-scale image
        # aka CSK tracker
        else:
            self.interp_factor = 0.075
            self.sigma = 0.2
            self.cell_size = 1
            self._hogfeatures = False

        # 是否开启多尺度，也就是在目标跟踪的过程中，适应目标大小的变化
        if multiscale:
            # 模板大小，在计算 _tmpl_sz 时，较大边长被归一成 96，而较小的边按比例缩小
            self.template_size = 96   # template size
            # 多尺度估计时的尺度步长，最后形成的尺度池为：{1 / 1.05, 1, 1.05}，对于每一帧图像，检测的时候
            # 会计算三个响应值:
            # 1.将原图像缩小，即乘以 1 / 1.05，计算一次响应值
            # 2.使用原图像计算一次响应值
            # 3.将原图像扩大，即乘以 1.05，计算一次响应值
            # 最后取响应值最大的图像尺寸作为检测到的图像
            # scale step for multi-scale estimation
            self.scale_step = 1.05
            # to down weight detection scores of other scales for added stability
            self.scale_weight = 0.96
        elif fixed_window:
            self.template_size = 96
            self.scale_step = 1
        else:
            self.template_size = 1
            self.scale_step = 1

        self._tmpl_sz = [0, 0]  # cv::Size, [width, height]  #[int,int]
        self._roi = [0., 0., 0., 0.]  # cv::Rect2f, [x,y,width,height]  #[float,float,float,float]
        self.size_patch = [0, 0, 0]  # [int,int,int]
        self._scale = 1.   # float

        # self._alphaf 即核相关滤波的 alpha 参数，需要通过回归来进行训练
        self._alphaf = None
        # self._prob 是初始化时制作的高斯响应图，可以看成是标签
        self._prob = None
        # self._tmpl 是 fhog 特征矩阵，shape 为 [numFeatures, sizeY * sizeX]，其中 numFeatures 一般为 31
        # fhog 相对于 hog 多了邻域归一化和 PCA 降维的过程
        self._tmpl = None
        self.hann = None

    # 使用第一帧和它的跟踪框，初始化 KCF 跟踪器
    def init(self, roi, image):
        self._roi = list(map(float, roi))
        assert (roi[2] > 0 and roi[3] > 0)
        # _tmpl 提取的 fhog 特征矩阵，shape 为 [numFeatures, sizeY * sizeX]
        self._tmpl = self.getFeatures(image, self._roi, 1)
        # _prob 是初始化时的高斯响应图，并且 size_patch 的大小为 [sizeY, sizeX, numFeatures]
        self._prob = self.createGaussianPeak(self.size_patch[0], self.size_patch[1])
        # _alphaf 是频域中的核相关滤波参数，有两个通道分成实部和虚部，初始化为 [sizeY, sizeX, 2] 的 shape
        # 可以说求出了 _alphaf 参数，就可以求出核相关滤波器，用于对跟踪目标进行检测
        self._alphaf = np.zeros((self.size_patch[0], self.size_patch[1], 2), np.float32)
        # 利用提取得到的 fhog 特征进行回归训练，得到 alpha 参数
        self.train(self._tmpl, 1.0)

    # 初始化 hanning 窗口，函数只在第一帧被执行
    # 目的是采样时为不同的样本分配不同的权重，0.5 * 0.5 是用汉宁窗归一化为 [0, 1]，得到的矩阵值就是每个样本的权重
    def createHanningMats(self):
        hann2t, hann1t = np.ogrid[0:self.size_patch[0], 0:self.size_patch[1]]

        hann1t = 0.5 * (1 - np.cos(2 * np.pi * hann1t / (self.size_patch[1] - 1)))
        hann2t = 0.5 * (1 - np.cos(2 * np.pi * hann2t / (self.size_patch[0] - 1)))
        hann2d = hann2t * hann1t

        if self._hogfeatures:
            hann1d = hann2d.reshape(self.size_patch[0] * self.size_patch[1])
            self.hann = np.zeros((self.size_patch[2], 1), np.float32) + hann1d
        # 相当于把 1d 的汉宁窗赋值成多个通道
        else:
            self.hann = hann2d
        self.hann = self.hann.astype(np.float32)

    # 标签制作，函数只在第一帧的时候执行（高斯响应）
    # 对于 ground_truth，由于模板函数的中心就是目标框的中心，因此论文中使用高斯分布函数作为标签，其分布函数为:
    # g(x, y) = exp[(-1 / (2 * sigma * sigma)) * ((i - cx) ^ 2 + (j - cy)^2)]
    # sigma = sqrt(sizeX * sizeY) / (padding * output_sigma_factor)
    # 其中，(cx, cy) 表示图像特征矩阵中心，padding 表示扩展框相对于目标框的变化比例为 2.5，output_sigma_factor 表示设定
    # 的一个值为 0.125
    def createGaussianPeak(self, sizey, sizex):
        # syh, sxh 为图像特征矩阵的中心点坐标
        syh, sxh = sizey / 2, sizex / 2
        output_sigma = np.sqrt(sizex * sizey) / self.padding * self.output_sigma_factor
        mult = -0.5 / (output_sigma * output_sigma)
        y, x = np.ogrid[0:sizey, 0:sizex]
        y, x = (y - syh) ** 2, (x - sxh) ** 2
        res = np.exp(mult * (y + x))
        return fftd(res)

    # 使用标准差为 sigma 的高斯卷积核函数，计算得到核相关矩阵。使用的公式就是多通道的高斯核函数
    # x1, x2 必须都是 m * n 的大小, m = sizeY, n = sizeX
    # 由于无论是训练还是检测都使用到了核相关矩阵，所以 hog 特征的融合主要是在这个过程中进行的，公式如下
    # k = exp(-d / (sigma * sigma))
    # d = max(0, (x1x1 + x2x2 - 2 * x1 * x2) / numel(x1))
    # numel(x1) 也就是 x1 的总像素点的个数
    def gaussianCorrelation(self, x1, x2):
        """
        我们首先把特征记为 x1[31]，x2[31]，是一个 31 维的容器，每个维度都是一个长度为 mn 的向量，也就是 x1 和 x2 的 shape 为 (31, mn)
        注意，这里的 m, n 也就是 size_patch[0] 和 size_patch[1]，所以 size_patch 的 shape 为 [m, n, 31]
        接下来第一步就是计算上面公式中的 x1 * x2
        1.首先分别对每个维度进行傅里叶变换，得到 xf1[31] 和 xf2[31]
        2.xf1[i] 以及 xf2[i] 表示是一个长度为 mn 的向量，所以将 xf1[i] 和 xf2[i] 转变为 [m,n] 的矩阵
        3.计算 xf1[i] 和 xf2[i] 的共轭在频域的点积（element-wise），这样得到的是 36 个 [m,n] 的复数矩阵，分别对每个矩阵都进行
        傅里叶逆变换得到 xf12[36], 是 36 个 [m,n] 的实数矩阵，然后把 36 个矩阵对应点求和得到一个矩阵记作 xf12，是一个 [m,n] 的实数矩阵
        """
        if self._hogfeatures:
            c = np.zeros((self.size_patch[0], self.size_patch[1]), np.float32)
            for i in range(self.size_patch[2]):
                # 将 x1[i], x2[i] 转变为 [m,n] 的矩阵
                x1aux = x1[i, :].reshape((self.size_patch[0], self.size_patch[1]))
                x2aux = x2[i, :].reshape((self.size_patch[0], self.size_patch[1]))
                # 傅立叶域点乘
                caux = cv2.mulSpectrums(fftd(x1aux), fftd(x2aux), 0, conjB=True)
                # 进行傅立叶逆变换
                caux = real(fftd(caux, True))
                c += caux
            c = rearrange(c)
        else:
            # 'conjB=' is necessary!
            c = cv2.mulSpectrums(fftd(x1), fftd(x2), 0, conjB=True)
            c = fftd(c, True)
            c = real(c)
            c = rearrange(c)

        if x1.ndim == 3 and x2.ndim == 3:
            d = (np.sum(x1[:, :, 0] * x1[:, :, 0]) + np.sum(x2[:, :, 0] * x2[:, :, 0]) - 2.0 * c) / (self.size_patch[0] * self.size_patch[1] * self.size_patch[2])
        elif x1.ndim == 2 and x2.ndim == 2:
            # python 中，* 表示矩阵中对应元素的点乘，而不是矩阵乘法
            # np.sum(x1 * x1) 相当于求矩阵 x1 的二范数，然后将矩阵中的每一个元素累加起来，得到的就是一个实数，对于 x2 也是同理
            d = (np.sum(x1 * x1) + np.sum(x2 * x2) - 2.0 * c) / (self.size_patch[0] * self.size_patch[1] * self.size_patch[2])

        # 等价于 d = max(d, 0)
        d = d * (d >= 0)
        # 得到核相关矩阵
        d = np.exp(-d / (self.sigma * self.sigma))

        return d

    def getFeatures(self, image, roi, inithann, scale_adjust=1.0):
        # self._roi 表示初始的目标框 [x, y, width, height]
        extracted_roi = [0, 0, 0, 0]
        # cx, cy 表示目标框中心点的 x 坐标和 y 坐标
        cx = roi[0] + roi[2] / 2  # float
        cy = roi[1] + roi[3] / 2  # float

        if inithann:
            # 保持初始目标框中心不变，将目标框的宽和高同时扩大相同倍数
            padded_w = roi[2] * self.padding
            padded_h = roi[3] * self.padding

            if self.template_size > 1:
                # 设定模板图像尺寸为 96，计算扩展框与模板图像尺寸的比例
                # 把最大的边缩小到 96，_scale 是缩小比例，_tmpl_sz 是滤波模板裁剪下来的 PATCH 大小
                # scale = max(w,h) / template
                self._scale = max(padded_h, padded_w) / float(self.template_size)
                # 同时将 scale 应用于宽和高，获取图像提取区域
                # roi_w_h = (w / scale, h / scale)
                self._tmpl_sz[0] = int(padded_w / self._scale)
                self._tmpl_sz[1] = int(padded_h / self._scale)
            else:
                self._tmpl_sz[0] = int(padded_w)
                self._tmpl_sz[1] = int(padded_h)
                self._scale = 1.

            if self._hogfeatures:
                # 由于后面提取 hog 特征时会以 cell 单元的形式提取，另外由于需要将频域直流分量移动到图像中心，因此需保证图像大小为 cell大小的偶数倍，
                # 另外，在 hog 特征的降维的过程中是忽略边界 cell 的，所以还要再加上两倍的 cell 大小
                self._tmpl_sz[0] = int(self._tmpl_sz[0]) // (2 * self.cell_size) * 2 * self.cell_size + 2 * self.cell_size
                self._tmpl_sz[1] = int(self._tmpl_sz[1]) // (2 * self.cell_size) * 2 * self.cell_size + 2 * self.cell_size
            else:
                self._tmpl_sz[0] = int(self._tmpl_sz[0]) // 2 * 2
                self._tmpl_sz[1] = int(self._tmpl_sz[1]) // 2 * 2

        # 选取从原图中扣下的图片位置大小
        extracted_roi[2] = int(scale_adjust * self._scale * self._tmpl_sz[0])
        extracted_roi[3] = int(scale_adjust * self._scale * self._tmpl_sz[1])
        extracted_roi[0] = int(cx - extracted_roi[2] / 2)
        extracted_roi[1] = int(cy - extracted_roi[3] / 2)

        # z 是当前被裁剪下来的搜索区域
        z = subwindow(image, extracted_roi, cv2.BORDER_REPLICATE)
        if z.shape[1] != self._tmpl_sz[0] or z.shape[0] != self._tmpl_sz[1]:
            z = cv2.resize(z, tuple(self._tmpl_sz))

        if self._hogfeatures:
            mapp = {'sizeX': 0, 'sizeY': 0, 'numFeatures': 0, 'map': 0}
            # 对目标图像进行处理，获取到方向梯度直方图，也就是 mapp['map']，mapp['map'] 的 shape 为 [sizeY, sizeX, 27]
            mapp = fhog.getFeatureMaps(z, self.cell_size, mapp)
            # 对目标图像的 cell 进行邻域归一化以及截断操作，得到的特征矩阵的 shape 为 [sizeY, sizeX, 108]，每一个 cell 的维度为 108 维
            mapp = fhog.normalizeAndTruncate(mapp, 0.2)
            # 对目标图像进行 PCA 降维，将每一个 cell 的维度由 108 维变为 27 + 4 = 31 维，得到的特征矩阵的 shape 为 [sizeY, sizeX, 31]
            mapp = fhog.PCAFeatureMaps(mapp)
            # size_patch 为列表，保存裁剪下来的特征图的 [sizeY，sizeX，numFeatures]，numFeatures 是 31 维（由 108 维降维成 31 维）
            self.size_patch = list(map(int, [mapp['sizeY'], mapp['sizeX'], mapp['numFeatures']]))
            # FeaturesMap 的 shape 为 [numFeatures, sizeY * sizeX]
            FeaturesMap = mapp['map'].reshape((self.size_patch[0] * self.size_patch[1], self.size_patch[2])).T   # (size_patch[2], size_patch[0]*size_patch[1])

        # 将 RGB 图像转变为单通道灰度图像
        else:
            if z.ndim == 3 and z.shape[2] == 3:
                FeaturesMap = cv2.cvtColor(z, cv2.COLOR_BGR2GRAY)   # z:(size_patch[0], size_patch[1], 3)  FeaturesMap:(size_patch[0], size_patch[1])   #np.int8  #0~255
            elif z.ndim == 2:
                FeaturesMap = z  # (size_patch[0], size_patch[1]) # np.int8  #0~255
            FeaturesMap = FeaturesMap.astype(np.float32) / 255.0 - 0.5
            # size_patch 为列表，保存裁剪下来的特征图的 [长，宽，1]
            self.size_patch = [z.shape[0], z.shape[1], 1]

        if inithann:
            self.createHanningMats()  # create Hanning Mats need size_patch

        # 加汉宁窗减少频谱泄漏
        FeaturesMap = self.hann * FeaturesMap

        cv2.imwrite('featuresMap.jpg', FeaturesMap)

        return FeaturesMap

    # 根据上一帧结果计算当前帧的目标位置
    # z 是前一帧的训练 / 第一帧的初始化结果，x 是当前帧当前尺度下的特征，peak_value 是检测结果峰值
    def detect(self, z, x):
        k = self.gaussianCorrelation(x, z)
        # 得到响应图
        # 这里实际上就是快速检测公式
        # F(z) = F(kzx) * F(alpha)，其中 * 表示对位相乘，并且利用转置消除共轭
        res = real(fftd(complexMultiplication(self._alphaf, fftd(k)), True))

        # pv:响应最大值，pi:相应最大点的索引数组: (列下标，行下标)
        _, pv, _, pi = cv2.minMaxLoc(res)   # pv:float  pi:tuple of int
        # 得到响应最大的点索引的 float 表示
        p = [float(pi[0]), float(pi[1])]   # cv::Point2f, [x,y]  #[float,float]

        # 使用幅值作差来定位峰值的位置
        # 也就是对于该响应矩阵，找出其最大响应值 peak_value 和最大响应位置 pxy，如果最大响应位置不在图像边界，那么
        # 分别比较最大响应位置两侧的响应大小，如果右侧比左侧高，或者下侧比上侧高，则分别将最大响应位置向较大的一侧移动一段距离
        # px = px + 0.5 * ((right - left) / (2 * peak_value - right - left))
        # py = py + 0.5 * ((down - up) / (2 * peak_value - down - up))
        if 0 < pi[0] < res.shape[1] - 1:
            p[0] += subPixelPeak(res[pi[1], pi[0] - 1], pv, res[pi[1], pi[0] + 1])
        if 0 < pi[1] < res.shape[0] - 1:
            p[1] += subPixelPeak(res[pi[1] - 1, pi[0]], pv, res[pi[1] + 1, pi[0]])

        # 得出偏离采样中心的位移，res.shape[1] / 2 表示采样中心的 x 坐标，res.shape[0] / 2 表示采样中心的 y 坐标
        p[0] -= res.shape[1] / 2.
        p[1] -= res.shape[0] / 2.

        # 返回偏离采样中心的位移和峰值
        return p, pv

    # 使用当前图像的检测结果进行训练，其实就是 KCF 算法中的核相关滤波训练/回归，得到 alpha 参数
    # x 是当前帧当前尺度下的 fhog 特征矩阵，shape 为 [numFeatures, sizeY * sizeX]）
    # train_interp_factor 是 interp_factor，可以看成是学习率，在初始化的时候，为 1；而在之后，则默认为 0.012
    def train(self, x, train_interp_factor):
        # alpha 是频域中的核相关滤波参数，有两个通道分别为实部和虚部
        k = self.gaussianCorrelation(x, x)

        # _prob 是初始化时的高斯响应图，相当于 y
        # 岭回归中的 lambdar 常数，正则化
        # 相当于 KCF 中的公式：F(y) / (F(kxx) + lambdar)
        alpha = complexDivision(self._prob, fftd(k) + self.lambdar)

        # 特征矩阵更新: template = (1 - 0.012) * template + 0.012 * x
        # self._tmpl 是 fhog 特征矩阵，shape 为 (sizeY, sizeX, 31)
        self._tmpl = (1 - train_interp_factor) * self._tmpl + train_interp_factor * x

        # _alphaf 是频域中核相关滤波参数的加权平均
        # alpha = (1 - 0.012) * alpha + 0.012 * alpha_x_z
        self._alphaf = (1 - train_interp_factor) * self._alphaf + train_interp_factor * alpha

    # 获取当前帧的目标位置以及尺度，image 为当前帧的整幅图像
    # 基于当前帧更新目标位置
    def update(self, image, roi):
        # 修正边界
        if roi[0] + roi[2] <= 0:
            roi[0] = -roi[2] + 1
        if roi[1] + roi[3] <= 0:
            roi[1] = -roi[2] + 1
        if roi[0] >= image.shape[1] - 1:
            roi[0] = image.shape[1] - 2
        if roi[1] >= image.shape[0] - 1:
            roi[1] = image.shape[0] - 2

        # 跟踪框、尺度框的中心
        cx = roi[0] + roi[2] / 2.
        cy = roi[1] + roi[3] / 2.

        # 尺度不变时检测峰值结果
        loc, peak_value = self.detect(self._tmpl, self.getFeatures(image, roi, 0, 1.0))

        # 略大尺度和略小尺度进行检测，也就是使用尺度池中的 1 / 1.05 和 1.05 两个尺度因子来对原图像进行缩小和放大
        if self.scale_step != 1:
            # Test at a smaller _scale
            new_loc1, new_peak_value1 = self.detect(self._tmpl, self.getFeatures(image, roi, 0, 1.0 / self.scale_step))
            # Test at a bigger _scale
            new_loc2, new_peak_value2 = self.detect(self._tmpl, self.getFeatures(image, roi, 0, self.scale_step))

            # 对于不同的尺度，都有着尺度惩罚系数 scale_weight，用此系数乘以该尺度下的最大响应值作为该尺度下的真实最大响应值
            # 取最大响应值对应的尺度为最佳尺度，也就是 self._scale
            if self.scale_weight * new_peak_value1 > peak_value and new_peak_value1 > new_peak_value2:
                loc = new_loc1
                peak_value = new_peak_value1
                self._scale /= self.scale_step
                roi[2] /= self.scale_step
                roi[3] /= self.scale_step
            elif self.scale_weight * new_peak_value2 > peak_value:
                loc = new_loc2
                peak_value = new_peak_value2
                self._scale *= self.scale_step
                roi[2] *= self.scale_step
                roi[3] *= self.scale_step

        roi[0] = cx - roi[2] / 2.0 + loc[0] * self.cell_size * self._scale
        roi[1] = cy - roi[3] / 2.0 + loc[1] * self.cell_size * self._scale

        if roi[0] >= image.shape[1] - 1:
            roi[0] = image.shape[1] - 1
        if roi[1] >= image.shape[0] - 1:
            roi[1] = image.shape[0] - 1
        if roi[0] + roi[2] <= 0:
            roi[0] = -roi[2] + 2
        if roi[1] + roi[3] <= 0:
            roi[1] = -roi[3] + 2

        assert(roi[2] > 0 and roi[3] > 0)

        return roi, peak_value

    def retrain(self, image, roi):
        # 使用当前的检测框来训练样本参数
        x = self.getFeatures(image, roi, 0, 1.0)
        self.train(x, self.interp_factor)

    def get_tmpl(self):
        return self._tmpl

def extract_image(image, roi):
    img = image[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
    cv2.imwrite('img.jpg', img)