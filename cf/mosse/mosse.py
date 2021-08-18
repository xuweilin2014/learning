import numpy as np
import cv2
import os
from utils import linear_mapping, pre_process, random_warp

"""
This module implements the basic correlation filter based tracking algorithm -- MOSSE
基于滤波的跟踪就是用在模板图片上训练好的滤波器去对目标物体的外表建模。目标最初是基于以第一帧中的目标
为中心的一个小跟踪窗口来选择的。从这点上来说，跟踪器和滤波器训练是一起进行的。通过在下一帧图片的搜索
窗口中去进行滤波来跟踪目标。滤波之后产生的最大值的地方就是目标的新位置。根据得到的新位置完成在线更新。

Correlation Filter应用于tracking方面最朴素的想法就是：相关是衡量两个信号相似值的度量，如果两个信号越相似，
那么其相关值就越高，而在tracking的应用里，就是需要设计一个滤波模板，使得当它作用在跟踪目标上时，得到的响应最大.

相关滤波的意思就是现在在第一帧图像中框选了一个目标，然后对这个目标训练一个滤波器（大小相同）使得其输出响应 g 在中间值最大。
其中输入图像给定，响应图也是可以直接生成的，一般都是用高斯函数，中间值最大，旁边逐渐降低。

MOSSE 算法的流程如下:
1.第一帧中以目标对象为中心截取一个目标窗口 [width, height]，获得目标在第一帧位置数据
2.生成一个和原图像大小相同的二维高斯响应矩阵，然后截取上面所选定的目标区域，然后进行快速傅立叶变换，得到响应 G
3.对第一帧，更新过滤器参数 Ai, Bi
4.读取第 n 帧图像，以第 n - 1 帧目标位置信息截取输入图像得到 fi
5.对 fi 进行快速傅立叶变换得到 Fi，将第 n - 1 帧中更新得到的过滤器 Ai 和 Bi 相除得到过滤器 Hi，将 Hi 和 Fi 进行
点乘，得到实际卷积输出 Gi，然后再对 Gi 进行反傅立叶变换得到实际输出 g
6.g 中的最大值位置也就是当前第 n 帧中目标所在的位置，更新当前帧中目标区域的位置 clip_pos
7.使用更新后的目标区域的位置重新进行截取，得到新的 fi
8.使用新的 Fi 和 G 来更新过滤器参数 Ai, Bi
9.重复 4-8 步，直到程序结束
"""


class Mosse:
    def __init__(self, args, img_path):
        # get arguments..
        self.args = args
        self.img_path = img_path
        # get the img lists...
        self.frame_lists = self._get_img_lists(self.img_path)
        self.frame_lists.sort()

    # start to do the object tracking...
    def start_tracking(self):
        # get the image of the first frame... (read as gray scale image...)
        # 读取到初始的第一帧图像，然后将图像由 BGR 转变为 GRAY 图像
        init_img = cv2.imread(self.frame_lists[0])
        init_frame = cv2.cvtColor(init_img, cv2.COLOR_BGR2GRAY)
        init_frame = init_frame.astype(np.float32)

        # get the init ground truth.. [x, y, width, height]
        # 这里通过手工框出想要选择的目标区域 [x, y, width, height]，其中 x 和 y 表示的是目标区域左上角顶点的坐标
        init_gt = cv2.selectROI('demo', init_img, False, False)
        init_gt = np.array(init_gt).astype(np.int64)

        # start to draw the gaussian response...
        # 得到高斯响应图（输入原始图像以及目标区域的位置 [x, y, width, height]）返回高斯函数矩阵，在选定的目标框的中心，其值最大
        response_map = self._get_gauss_response(init_frame, init_gt)

        # start to create the training set ...
        # get the goal...
        # 抽取高斯响应矩阵，矩阵的大小和选中的 ROI 的大小相同。
        g = response_map[init_gt[1]:init_gt[1] + init_gt[3], init_gt[0]:init_gt[0] + init_gt[2]]
        # 抽取目标区域的图像
        fi = init_frame[init_gt[1]:init_gt[1] + init_gt[3], init_gt[0]:init_gt[0] + init_gt[2]]
        # 对目标区域的高斯响应图做快速傅立叶变换
        G = np.fft.fft2(g)

        # 做滤波器的预训练
        # start to do the pre-training...
        Ai, Bi = self._pre_training(fi, G)

        # start the tracking...
        for idx in range(len(self.frame_lists)):

            current_frame = cv2.imread(self.frame_lists[idx])
            frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            frame_gray = frame_gray.astype(np.float32)

            if idx == 0:
                Ai = self.args.lr * Ai
                Bi = self.args.lr * Bi
                pos = init_gt.copy()
                # pos 的内容是 [leftX, topY, roi width, roi height]
                clip_pos = np.array([pos[0], pos[1], pos[0] + pos[2], pos[1] + pos[3]]).astype(np.int64)
            else:
                '''
                在当前帧中，使用上一帧更新后的搜索区域 (clip_pos) 在本帧中截取相同的位置，使用过滤器与截取区域执行相关操作
                相关性最大的位置就是响应最大值的位置，然后更新过滤器 (Ai, Bi)，更新搜索区域 (clip_pos)。
                '''

                # Ai 和 Bi 在上一帧中已经更新了，现在重新计算出滤波模板 Hi
                Hi = Ai / Bi
                fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
                fi = pre_process(cv2.resize(fi, (init_gt[2], init_gt[3])))

                # 使用 Hi 和 fi 进行卷积操作，得到一个响应矩阵
                Gi = Hi * np.fft.fft2(fi)
                # 对于频域下的 Gi 进行逆傅立叶变换得到实际的 gi
                gi = linear_mapping(np.fft.ifft2(Gi))

                # 找到响应矩阵 gi 中的最大值
                max_value = np.max(gi)
                # 获取到 gi 中最大值的坐标，这个位置就是当前帧中被跟踪目标的坐标，只不过这个坐标是相对于 gi，也就是目标区域而言的
                max_pos = np.where(gi == max_value)
                # gi.shape[0] / 2 就是上一个目标的 y 坐标，也是相对于 gi 这个区域而言，相减得到的 dy 就是当前目标与上一个目标在 y 方向的偏移量
                dy = int(np.mean(max_pos[0]) - gi.shape[0] / 2)
                # gi.shape[1] / 2 就是上一个目标的 x 坐标，也是相对于 gi 这个区域而言，相减得到的 dx 就是当前目标与上一个目标在 x 方向的偏移量
                dx = int(np.mean(max_pos[1]) - gi.shape[1] / 2)

                # update the position...
                # pos 的内容是 [leftX, topY, roi width, roi height]，也就是 roi 目标框左上角的坐标与目标框的宽 width 和高 height
                # 这里只是单纯的将 roi 目标框左上角的坐标进行移动，而对 roi 的长和宽不进行修改，因此 mosse 滤波无法处理跟踪目标的大
                # 小发生变化的情况
                pos[0] = pos[0] + dx
                pos[1] = pos[1] + dy

                # trying to get the clipped position [xmin, ymin, xmax, ymax]
                # clip_pos 表示的是在这一帧中，目标区域的新位置 [leftX, topY, rightX, bottomY]
                clip_pos[0] = np.clip(pos[0], 0, current_frame.shape[1])
                clip_pos[1] = np.clip(pos[1], 0, current_frame.shape[0])
                clip_pos[2] = np.clip(pos[0] + pos[2], 0, current_frame.shape[1])
                clip_pos[3] = np.clip(pos[1] + pos[3], 0, current_frame.shape[0])
                clip_pos = clip_pos.astype(np.int64)

                # get the current fi..
                fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
                fi = pre_process(cv2.resize(fi, (init_gt[2], init_gt[3])))

                # online update...
                # 在线更新 Ai, Bi
                # 这里的 lr 就是 learning rate，学习率，加入 lr 可以使得模型更加重视最近的帧，并且使得先前的帧的效果随时间衰减
                Ai = self.args.lr * (G * np.conjugate(np.fft.fft2(fi))) + (1 - self.args.lr) * Ai
                Bi = self.args.lr * (np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))) + (1 - self.args.lr) * Bi

            # visualize the tracking process...
            cv2.rectangle(current_frame, (pos[0], pos[1]), (pos[0] + pos[2], pos[1] + pos[3]), (0, 0, 255), 2)
            cv2.imshow('demo', current_frame)
            cv2.waitKey(100)

            # if record... save the frames..
            if self.args.record:
                frame_path = 'record_frames/' + self.img_path.split('/')[1] + '/'
                if not os.path.exists(frame_path):
                    os.mkdir(frame_path)
                cv2.imwrite(frame_path + str(idx).zfill(5) + '.png', current_frame)

    # pre train the filter on the first frame...
    def _pre_training(self, init_frame, G):
        # G 的大小就是选中的目标区域的大小
        height, width = G.shape
        fi = cv2.resize(init_frame, (width, height))
        # pre-process img..
        fi = pre_process(fi)

        # np.fft.fft2 表示求 fi 的傅立叶变换
        # np.conjugate 表示求矩阵的共轭
        # 比如 g = np.matrix('[1+2j, 2+3j; 3-2j, 1-4j]')
        # g.conjugate 为 matrix([[1-2j, 2-3j],[3+2j, 1+4j]])
        Ai = G * np.conjugate(np.fft.fft2(fi))
        Bi = np.fft.fft2(init_frame) * np.conjugate(np.fft.fft2(init_frame))

        # 对 fi 进行多次刚性形变，增强检测的鲁棒性，计算出 Ai 和 Bi 的初始值
        for _ in range(self.args.num_pretrain):
            if self.args.rotate:
                fi = pre_process(random_warp(init_frame))
            else:
                fi = pre_process(init_frame)
            Ai = Ai + G * np.conjugate(np.fft.fft2(fi))
            Bi = Bi + np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))

        return Ai, Bi

    # get the ground-truth gaussian response...
    def _get_gauss_response(self, img, gt):
        # get the shape of the image..
        height, width = img.shape
        # get the mesh grid...
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))

        # get the center of the object...
        # 得到选定的目标区域的中心点坐标
        center_x = gt[0] + 0.5 * gt[2]
        center_y = gt[1] + 0.5 * gt[3]

        # cal the distance...
        # 创建一个以选定的目标中点为中心，且符合二维高斯分布的响应矩阵，矩阵大小等于原图像 img 的大小
        # 原始的二维高斯函数中，方差有两个: sigmaX 和 sigmaY，其中 sigmaX 为 x 方向的方差，sigmaY 为 y 方向的方差
        # 不过这里取相同的值，使得二维高斯模型在平面上的投影就是一个圆形，意思是与目标中心 (x0, y0) 的距离一样的点的权重是一样的，
        # 如果取不一样的值，那么投影为一个椭圆形，距离目标中心会得到不一样的权重
        exponent = (np.square(xx - center_x) + np.square(yy - center_y)) / (2 * self.args.sigma)
        # get the response map...
        # 获取到高斯响应矩阵
        response = np.exp(-exponent)

        # normalize...
        # 对响应矩阵进行归一化处理: (x - min) / (max - min)
        response = linear_mapping(response)
        return response

    # it will extract the image list 
    @staticmethod
    def _get_img_lists(img_path):
        frame_list = []
        for frame in os.listdir(img_path):
            if os.path.splitext(frame)[1] == '.jpg':
                frame_list.append(os.path.join(img_path, frame))
        return frame_list

    # it will get the first ground truth of the video..
    @staticmethod
    def _get_init_ground_truth(img_path):
        gt_path = os.path.join(img_path, 'groundtruth.txt')
        with open(gt_path, 'r') as f:
            # just read the first frame...
            line = f.readline()
            gt_pos = line.split(',')

        return [float(element) for element in gt_pos]
