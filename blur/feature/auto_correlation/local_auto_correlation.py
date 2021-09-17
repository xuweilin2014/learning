import cv2
import matplotlib.pyplot as plt
import numpy as np

EPSILON = 1.19209e-06


def local_auto_correlation(img, ksize=3, bins=36):
    """
    使用自相关函数来表示一个信号在间隔一段时间之后和这个信号本身的相似程度，而运动模糊图像沿着模糊方向的自相关
    函数值比较小，而沿着垂直于模糊方向的自相关函数值比较大。因此可以使用这个特性来区分运动模糊和清晰图像

    params: img:灰度图片
            ksize：Sobel算子窗口大小

    return：corner：与源图像一样大小，角点处像素值设置为255
    """

    # 响应函数 R 中的系数 k
    k = 0.04
    # 设定阈值
    threshold = 0.01
    # 是否非极大值抑制
    WITH_NMS = False

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
    # 对于图像中的每一点 (x,y) 都会生成一个矩阵 M: [[IxIx, IxIy], [IxIy, IyIy]]
    m = [np.array([[m[i, j, 0], m[i, j, 2]], [m[i, j, 2], m[i, j, 1]]]) for i in range(h) for j in range(w)]

    # 4、对于图像中的每一个点都会生成一个矩阵 M，接着就会求出这个矩阵 M 的特征值 (v1, v2) 和特征向量，选取小一点的特征值对应的
    # 特征向量，然后根据这个特征向量的方向加上权重 sqrt(max(v1, v2) / min(v1, v2)) 得到直方图，最后使用 36 维（默认）的
    # 直方图作为特征值
    hist = np.zeros(bins)

    counter = 0
    for matrix in m:
        # 求解矩阵 M 得到对应的特征值 (v1, v2) 和特征向量
        eigen_values, eigen_vectors = np.linalg.eig(matrix)
        # 选取较小的特征值对应的特征向量
        vectors = eigen_vectors[np.argmin(eigen_values)]

        if np.min(eigen_values) <= EPSILON:
            continue

        counter += 1
        # 计算权重 sqrt(max(v1, v2) / min(v1, v2))
        weight = np.sqrt(np.max(eigen_values) / np.min(eigen_values))

        index = 0
        # 特征向量 vector 的方向映射到 (0, 180) 这个区间
        if vectors[0] * vectors[1] > 0:
            theta = (np.arctan(vectors[1] / vectors[0]) / np.pi) * 180
            index = int((theta / 180) * bins - 1)
            hist[index] += weight
        elif vectors[0] * vectors[1] < 0:
            theta = 180 - (np.arctan(np.abs(vectors[1] / vectors[0]))) * 180
            index = int(theta / 180 * bins - 1)
            hist[index] += weight

    counter = 1 if counter == 0 else counter
    # 对得到的直方图向量 hist 进行标准化，最后就得到了特征向量 hist
    if np.sum(hist) > EPSILON:
        hist /= np.sum(hist)

    hist[np.isnan(hist).astype('int')] = np.min(hist[(~np.isnan(hist)).astype('int')])
    return hist

def correlation_blur_feature_plot(img_path, blur_img_path, bins=36, patch_size=11):
    img_blur = cv2.imread(blur_img_path, 0)
    img_blur = (img_blur - np.min(img_blur)) / (np.max(img_blur) - np.min(img_blur))
    hist_blur = local_auto_correlation(img_blur, bins=bins)

    img = cv2.imread(img_path, 0)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    hist_no_blur = local_auto_correlation(img, bins=bins)

    x = np.linspace(1, bins, bins)
    plt.plot(x, hist_blur, label='blur', color='red', marker='o', linestyle='-')
    plt.plot(x, hist_no_blur, label='no blur', color='blue', marker='o', linestyle='-')
    plt.legend()
    plt.show()

# noinspection PyShadowingNames
def im2col(img, patch_size):
    img_shape = img.shape
    sh = img_shape[0] - patch_size[0] + 1
    sw = img_shape[1] - patch_size[1] + 1
    # 设 A 为 m * n，如果划分为 [p, q] 的块，最后的矩阵的行数为 p * q，列数为 (m - p + 1)*(n - q + 1)
    result = np.empty((patch_size[0] * patch_size[1], sh * sw))
    for i in range(sw):
        for j in range(sh):
            result[:, i * sh + j] = img[j:j + patch_size[0], i:i + patch_size[1]].ravel(order='F')

    return result

if __name__ == '__main__':
    correlation_blur_feature_plot('../input/356.jpg', '../input/356_blur.jpg')
    correlation_blur_feature_plot('../input/race.jpeg', '../input/race_60.jpg')
    correlation_blur_feature_plot('../input/boat.jpeg', '../input/boat_35.jpeg')
    correlation_blur_feature_plot('../input/449.jpg', '../input/449_blur.jpg')
