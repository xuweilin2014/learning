from time import time
import cv2
import numpy as np
from numba import jit

"""
实现了 Mean-Shift Tracker 算法，可以对彩色图像以及灰度图像中的目标进行追踪
参考了 mean-shift.pdf 里面的算法流程以及下面的 matlab 代码
https://github.com/mohitkumarahuja/Visual-Tracking-Using-MeanShift
"""

EPSILON = 1.19209e-09
ITERATION_LIMIT = 200
BIN_NUM = 8

# noinspection PyAttributeOutsideInit,PyUnusedLocal,PyShadowingNames
class MeanShiftTracker:

    def init(self, frame, rect):
        self.rect = np.round(rect)
        x, y, w, h = self.rect
        roi_center = np.round(np.array([x + w / 2, y + h / 2]))
        self.roi_width = w
        self.roi_height = h
        self.bins = BIN_NUM

        if len(frame.shape) == 2:
            frame = np.reshape(frame, frame.shape + (1,))

        self.VIDEO_HEIGHT, self.VIDEO_WIDTH, self.channels = frame.shape

        '''
        FIRST, YOU NEED TO DEFINE THE COLOR MODEL OF THE OBJECT
        '''
        # compute target object color probability distribution given the center and size of the ROI
        img_patch = self.extract_image_patch(frame, roi_center, w, h)
        # color distribution (color histogram) in RGB color space
        self.target_model = self.color_distribution(img_patch)
        # set the location to the previous one
        self.prev_center = roi_center

    def track(self, frame):
        if len(frame.shape) == 2:
            frame = np.reshape(frame, frame.shape + (1,))

        while True:
            '''STEP 1'''
            roi_width, roi_height = self.roi_width, self.roi_height
            # calculate the pdf of the previous position
            img_patch = self.extract_image_patch(frame, self.prev_center, self.roi_width, self.roi_height)
            color_model = self.color_distribution(img_patch)
            # evaluate the Bhattacharyya coefficient
            rho_0 = self.compute_bhattacharyya_coef(self.target_model, color_model)

            '''STEP 2'''
            # derive the weights
            weights = self.compute_weights(img_patch, self.target_model, color_model)

            '''STEP 3'''
            # compute the mean-shift vector, using Epanechnikov kernel,
            # it reduces to a weighted average
            z = self.compute_meanshift_vector(img_patch, weights)

            # check if the bhattacharyya coeff is higher with the new center, if not, take the half between both centers
            new_img_patch = self.extract_image_patch(frame, z, roi_width, roi_height)
            new_color_model = self.color_distribution(new_img_patch)
            new_rho_0 = self.compute_bhattacharyya_coef(self.target_model, new_color_model)

            rho_counter = 1

            while new_rho_0 < rho_0 and rho_counter <= ITERATION_LIMIT:
                rho_counter += 1
                z = 0.5 * (self.prev_center + z)
                new_img_patch = self.extract_image_patch(frame, z, roi_width, roi_height)
                new_color_model = self.color_distribution(new_img_patch)
                new_rho_0 = self.compute_bhattacharyya_coef(self.target_model, new_color_model)

            self.new_center = z

            '''STEP 4'''
            if np.linalg.norm(self.new_center - self.prev_center, ord=1) < 1e-5:
                self.prev_center = self.new_center
                break

            self.prev_center = self.new_center

        VIDEO_HEIGHT, VIDEO_WIDTH = self.VIDEO_HEIGHT, self.VIDEO_WIDTH
        ix = np.round(max(self.prev_center[0] - roi_width / 2, 1))
        iy = np.round(max(self.prev_center[1] - roi_height / 2, 1))
        ex = np.round(min(VIDEO_WIDTH, ix + roi_width + 1))
        ey = np.round(min(VIDEO_HEIGHT, iy + roi_height + 1))

        return np.array([ix, iy, ex, ey])

    def extract_image_patch(self, image, center, width, height):
        """
        this function extract an image patch in image I given the center and size of the ROI
        :param center: [center_x, center_y] of roi
        :param width:  roi_width
        :param height: roi_height
        :return:
        """
        VIDEO_HEIGHT, VIDEO_WIDTH = self.VIDEO_HEIGHT, self.VIDEO_WIDTH
        y = center[1] - height / 2
        x = center[0] - width / 2

        h2 = int(min(VIDEO_HEIGHT, y + height + 1))
        w2 = int(min(VIDEO_WIDTH, x + width + 1))
        h = int(max(y, 1))
        w = int(max(x, 1))

        return image[h:h2, w:w2, :]

    @jit(cache=True)
    def compute_weights(self, img_patch, q_target, p_current):
        """
        计算 img_patch 图像中每一个像素点的权重值
        :param img_patch: 要计算的 img_patch 图像
        :param q_target: 要跟踪的目标模型
        :param p_current: 当前的候选模型
        :return:
        """
        bins = self.bins
        h, w, c = img_patch.shape
        # compute the ratio vector between both color distribution
        ratio = np.sqrt(q_target / p_current)
        ratio[p_current <= EPSILON] = 0
        weights = np.zeros((h, w))

        # 1.find weights over all the patch, for every pixels
        # 2.check in which bin is the current pixel value
        # 3.compute the weight
        for k in range(c):
            indices = (np.divide(img_patch[:,:,k], 255) * bins).astype('int32')
            indices[indices == bins] = bins - 1
            weights += ratio[indices + k * bins]

        return weights / c

    def compute_meanshift_vector(self, img_patch, weights):
        h, w, c = img_patch.shape
        sz = np.floor(np.array([h, w]) / 2)

        # get array of coordinates
        xv, yv = np.linspace(1, w, w), np.linspace(1, h, h)
        x, y = np.meshgrid(xv, yv)
        # 这里减去 sz，就是公式中的减去候选区域中的中心点
        x = x - sz[1] + self.prev_center[0]
        y = y - sz[0] + self.prev_center[1]
        # 由于使用的是 Epanechnikov 核函数，而 g(x) = - dk(x) / dx = 1, ||x|| <= 1
        z = np.array([np.sum(x * weights), np.sum(y * weights)] / np.sum(weights))

        return np.round(z)

    @staticmethod
    def compute_bhattacharyya_coef(p, q):
        return sum(np.sqrt(p * q))

    @jit(cache=True)
    def color_distribution(self, img_patch):
        bins = self.bins
        h, w, c = img_patch.shape
        cd = np.zeros(bins * c)
        center = np.round(np.array([w / 2, h / 2]))
        dist = np.zeros((h, w))

        # compute distances
        # 计算 patch 中各点到 patch 中心点之间的距离
        xv, yv = np.linspace(1, w, w), np.linspace(1, h, h)
        x, y = np.meshgrid(xv, yv)
        dist = np.sqrt(np.power(x - center[0], 2) + np.power(y - center[1], 2))

        # normalize the distances
        dist = dist / np.max(dist)

        # build the histogram and weight with the kernel
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    # compute which bin is corresponding to the pixel value
                    index = np.int(img_patch[i, j, k] / 255 * bins)
                    index = index - 1 if index == 8 else index
                    # use Epachnikov kernel function
                    # 如果 ||x|| <= 1, k(x) = 1 - x
                    # 如果 ||x|| > 1, k(x) = 0
                    if dist[i, j] ** 2 < 1:
                        kE = 2 / np.pi * (1 - dist[i, j] ** 2)
                    else:
                        kE = 0
                    # add the kernel value to the bin
                    cd[k * bins + index] += kE

        # normalize the kernel value
        cd = cd / np.sum(cd)
        cd[cd <= EPSILON] = EPSILON

        return cd

selecting_object = False
init_tracking = False
on_tracking = False
ix, iy, cx, cy = -1, -1, -1, -1
w, h = 0, 0

interval = 1
duration = 0.01

def draw_bounding_box(event, x, y, flags, param):
    global selecting_object, init_tracking, on_tracking, ix, iy, cx, cy, w, h

    if event == cv2.EVENT_LBUTTONDOWN:
        selecting_object = True
        on_tracking = False
        ix, iy = x, y
        cx, cy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        cx, cy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        selecting_object = False
        if abs(x - ix) > 10 and abs(y - iy) > 10:
            w, h = abs(x - ix), abs(y - iy)
            ix, iy = min(x, ix), min(y, iy)
            init_tracking = True
        else:
            on_tracking = False

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    tracker = MeanShiftTracker()

    cv2.namedWindow('tracking')
    cv2.setMouseCallback('tracking', draw_bounding_box)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 180)
        if selecting_object:
            cv2.rectangle(frame, (ix, iy), (cx, cy), (0, 255, 255), 2)
        elif init_tracking:
            cv2.rectangle(frame, (ix, iy), (ix + w, iy + h), (0, 255, 255), 2)
            init_tracking = False
            on_tracking = True
            tracker.init(frame, np.array([ix, iy, w, h]))
        elif on_tracking:
            t0 = time()
            bounding_box = tracker.track(frame)
            t1 = time()

            bounding_box = list(map(int, bounding_box))
            # 画出物体的当前位置框
            cv2.rectangle(frame, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (0, 255, 255), 2)

            duration = 0.8 * duration + 0.2 * (t1 - t0)
            # duration = t1 - t0
            cv2.putText(frame, 'FPS: ' + str(1 / duration)[:4].strip('.'), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow('tracking', frame)
        c = cv2.waitKey(interval) & 0xFF
        if c == 27 or c == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
