import os
import time
import cv2

interval = 1
duration = 0.01

def track():
    tracker_types = ['CSRT', 'BOOSTING', 'MIL', 'TLD', 'MEDIANFLOW']
    for tracker_type in tracker_types:
        if not os.path.exists(tracker_type):
            os.mkdir(tracker_type)

        blur_seq_path = '../../blur/blur_seq/'
        for dir_name in os.listdir(blur_seq_path):
            print(dir_name)
            ground_truth_path = os.path.join(blur_seq_path, dir_name, 'groundtruth.txt')
            imgs_path = os.path.join(blur_seq_path, dir_name, 'img')

            tracker = select_tracker(tracker_type)

            with open(ground_truth_path, 'r') as f:
                init_pos = f.readline()[:-1].split(',')
                ix, iy, w, h = int(init_pos[0]), int(init_pos[1]), int(init_pos[2]), int(init_pos[3])

            output_path = os.path.join(tracker_type, dir_name + '.txt')

            with open(output_path, 'w+') as f:

                cv2.namedWindow('tracking')
                counter = 1
                init = True

                for img_name in os.listdir(imgs_path):
                    img_path = os.path.join(imgs_path, img_name)
                    frame = cv2.imread(img_path)
                    global duration

                    if init:
                        cv2.rectangle(frame, (ix, iy), (ix + w, iy + h), (0, 255, 255), 2)
                        f.write(str([ix, iy, w, h])[1:-1] + '\n')
                        # 初始化 kcf tracker，开始对目标进行跟踪
                        tracker.init(frame, (ix, iy, w, h))
                        init = False
                    else:
                        t0 = time.time()
                        _, boundingbox = tracker.update(frame)
                        t1 = time.time()

                        boundingbox = list(map(int, boundingbox))
                        print(boundingbox)
                        f.write(str(boundingbox)[1:-1] + '\n')
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

                cv2.destroyAllWindows()

def select_tracker(tracker_type):
    tracker = None

    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    elif tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    elif tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    elif tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    elif tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    elif tracker_type == 'CSRT':
        tracker = cv2.TrackerCSRT_create()

    return tracker


if __name__ == '__main__':
    track()
