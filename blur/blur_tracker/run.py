import os
import pickle
import time

import kcf
from angle_estimation import *
from local_kurtosis.local_kurtosis import *

selectingObject = False
initTracking = False
onTracking = False
ix, iy, cx, cy = -1, -1, -1, -1
w, h = 0, 0

interval = 1
duration = 0.01

def get_img_list(img_path):
    frame_list = []
    for frame in os.listdir(img_path):
        if os.path.splitext(frame)[1] == '.jpg':
            frame_list.append(os.path.join(img_path, frame))

    frame_list.sort()
    return frame_list


if __name__ == '__main__':

    print('================ 开始加载 SVM 模型 ====================')
    # 开始训练模糊特征分类器 svm，用来区分是否出现运动模糊
    with open('../feature/svm/model.pickle', 'rb') as f:
        clfier = pickle.load(f)
    print('===================== SVM 模型加载完毕 ================')

    print('================== 加载图片序列中 ===================')
    img_path = 'blur_seq/blur_car4/'
    frame_list = get_img_list(img_path)
    print('==================== 图片加载完成 ===================')

    # if you use hog feature, there will be a short pause after you draw a first bounding box, that is due to the use of Numba
    # hog, fixed_window, multiscale
    tracker = kcf.KCFTracker(True, True, True)
    init = True
    counter = 1
    cv2.namedWindow('tracking')

    print('================== 开始进行跟踪 =====================')
    for idx in range(len(frame_list)):
        frame = cv2.imread(frame_list[idx])

        if init:
            ix, iy, w, h = 197, 203, 170, 149
            cv2.rectangle(frame, (ix, iy), (ix + w, iy + h), (0, 255, 255), 2)
            print([ix, iy, w, h])
            # 初始化 kcf tracker，开始对目标进行跟踪
            tracker.init([ix, iy, w, h], frame, clfier)
            init = False
        else:
            t0 = time.time()

            boundingbox, _ = tracker.update(frame)
            t1 = time.time()

            boundingbox = list(map(int, boundingbox))
            print(boundingbox)
            # 画出物体的当前位置框
            cv2.rectangle(frame, (boundingbox[0], boundingbox[1]), (boundingbox[0] + boundingbox[2], boundingbox[1] + boundingbox[3]), (0, 255, 255), 3)

            duration = 0.8 * duration + 0.2 * (t1 - t0)
            # duration = t1 - t0
            cv2.putText(frame, 'FPS: ' + str(1 / duration)[:4].strip('.'), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, 'Frame: ' + str(counter), (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            counter += 1

        cv2.imshow('tracking', frame)
        c = cv2.waitKey(interval) & 0xFF
        if c == 27 or c == ord('q'):
            break

    print('=================== 跟踪结束 ======================')

    cv2.destroyAllWindows()
