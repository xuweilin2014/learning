import numpy as np
import cv2
import time

frame_to_start = 172
segment_merge_threshold = 3000
min_blob_size = 1200

current_frame = frame_to_start

gray = None
gray_prev = None

filename = "video/rouen_video.avi"
# filename = "video/atrium_video.avi"
# filename = "video/sherbrooke_video.avi"
# filename = "video/stmarc_video.avi"
# filename = "video/levesque_video.mov"

# 用来保存上一帧中的目标矩形框大小和坐标
rect_in_frame = []
# 用来保存当前帧中目标的 id，目标 id 由全局变量 obj_num 来生成
label_in_frame = []
# 当无匹配的帧数达到 8 时，就认为这个物体已经离开了画面
# delay_to_delete 列表中记录的就是各个 kcf tracker 没有匹配的帧数
delay_to_delete = []
# 当不同的 kcf tracker 在当前帧中互相遮挡时，也就是一个 COR 对应着多个 TO,
# 比如 kcf tracker[0], kcf tracker[2] 互相遮挡，在新一帧中，(COR)i 所占的面积等于 (TO)0、(TO)2
# 因此，我们需要把 group_whenOcclusion[0] = group_whenOcclusion[2] = Boundrect[i]
group_when_occlusion = []
# 用来保存每一个目标对应的 kcf tracker
tracker_vector = []
# 如果前一帧中没有一个目标的话，就将 prevNo_obj 设置为 1
prev_no_obj = 1
# 视频帧中的目标总数
obj_num = 0

show_msg = []

rect_bound_save = []
rect_frame_save = []
obj_appear_frame = []


def aoi_gravity_center(src, bound):
    x, y, w, h = bound
    sumx = num_pixel = sumy = 0
    ROI = src(bound)

    for x in range(ROI.cols):
        for y in range(ROI.rows):
            val = ROI[y, x]
            if val >= 50:
                sumx += x
                sumy += y
                num_pixel += 1

    px = sumx / num_pixel + x
    py = sumy / num_pixel + y
    return [px, py]


def centroid_close_enough(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) < segment_merge_threshold


def overlap_iou(bounding1, bounding2):
    x1, y1, w1, h1 = bounding1
    x2, y2, w2, h2 = bounding2

    if x1 > x2 + w2: return 0.0
    if y1 > y2 + h2: return 0.0
    if x1 + w1 < x2: return 0.0
    if y1 + h1 < y2: return 0.0

    col_int = min(x1 + w1, x2 + w2) - max(x1, x2)
    row_int = min(y1 + h1, y2 + h2) - max(y1, y2)
    intersection = col_int * row_int

    return intersection / (w1 * h1)


def create_new_object(frame, bound):
    global obj_num
    obj_num += 1
    save_label = str(obj_num)

    show_msg.append(save_label)
    label_in_frame.append(obj_num)
    rect_in_frame.append(bound)
    delay_to_delete.append(0)

    temp = [0, 0, 0, 0]
    group_when_occlusion.append(temp)

    tracker = cv2.TrackerKCF_create()
    x, y, w, h = bound
    tracker.init(frame, (x, y, w, h))
    tracker_vector.append(tracker)

    path_save = [bound]
    rect_bound_save.append(path_save)
    frame_save = [current_frame]
    rect_frame_save.append(frame_save)


def deliver_tracker(frame, index, bounding):
    tracker_vector.pop(index)
    rect_bound_save[index].pop(-1)
    rect_frame_save[index].pop(-1)

    tracker = cv2.TrackerKCF_create()
    tracker_vector.insert(index, tracker)
    tracker_vector[index].init(frame, bounding)
    rect_in_frame[index] = bounding
    rect_in_frame[index] = tracker_vector[index].update(frame)

    rect_frame_save[index].append(current_frame)
    rect_bound_save[index].append(rect_in_frame[index])
    cv2.rectangle(frame, rect_in_frame[index], (255, 0, 0), 2, 1)


# noinspection PyShadowingBuiltins
def delete_obj(find_tracker, add_this_frame, i):
    rect_in_frame.pop(i)
    label_in_frame.pop(i)
    delay_to_delete.pop(i)
    group_when_occlusion.pop(i)
    find_tracker.pop(i)
    tracker_vector.pop(i)
    show_msg.pop(i)
    rect_bound_save.pop(i)
    rect_frame_save.pop(i)
    add_this_frame.pop(i)


# noinspection PyShadowingBuiltins
def multiple_kcf_tracker(frame, bounding, centroid):
    global prev_no_obj
    # prev_no_obj 为 1 表示前一帧中没有物体
    if prev_no_obj == 1:
        for i in range(len(bounding)):
            # 前一帧中没有物体，因此为这一帧新出现的目标创建对应的跟踪器 kcf tracker
            # 并且创建两个一维列表，分别用来保存被跟踪的物体轨迹和物体的出现的帧数
            create_new_object(frame, bounding[i])
        prev_no_obj = 0
    else:
        # 如果 tracker_vector 的 size 为 0，表明前一帧中没有 tracker
        if len(tracker_vector) == 0:
            prev_no_obj = 1
            return

        # identify whether KCF tracker match in this frame or not
        # 往 find_tracker 向量最后插入 rect_in_frame.size() 个 -1，在最后会判断 find_tracker[i] 是否为 -1，
        # 如果为 -1 的话，那么就说明 kcf tracker 在当前帧中没有任何 COR 和其匹配
        find_tracker = [-1 for _ in range(len(rect_in_frame))]
        # identify whether it's property saved to xml or not
        add_this_frame = [0 for _ in range(len(rect_in_frame))]
        # calculate how many existing objects in Bounding Boxes of current frame.
        # kcf_num_blob 初始化为一个长度为 bounding.size() 并且全部为 0 的向量
        # kcf_num_blob[i] = j，表明当前帧中第 i 个目标框 (COR)i 和 j 个 kcf tracker 相匹配
        kcf_num_blob = [0 for _ in range(len(bounding))]

        # save match property for each existing objects
        # kcf_match 初始化为一个长度为 rect_in_frame.size() 并且全部为 0 的向量，表示 kcf tracker[i] 和当前帧中哪一个 BGS 相匹配
        # kcf_match[i] = j，表明第 i 个 kcf tracker 和当前帧中的 BGS 的第 j 个目标相匹配
        # kcf_match[i] = -1，表明第 i 个 kcf tracker 和当前帧中的任何 BGS 目标都不匹配
        kcf_match = [0 for _ in range(len(rect_in_frame))]

        # match objects by comparing overlapping rates of objects saved in previous frame
        # 通过计算前一帧中 kcf tracker 的 bbx 和当前帧中 BGS 的 bbx 的 iou 值，来进行匹配
        for i in range(len(rect_in_frame)):
            max = 0
            # label 就表示当前帧中和 kcf tracker[i] 最匹配（iou 值最高）的 bounding box 为 bounding[label]
            label = 0

            for j in range(len(bounding)):
                # find the most suitable one
                overlap = overlap_iou(rect_in_frame[i], bounding[j])
                if overlap > max:
                    max = overlap
                    label = j

            # object matches
            if max > 0:
                kcf_num_blob[label] += 1
                kcf_match[i] = label
            # object missing
            else:
                kcf_match[i] = -1

        # Occlusion occurs
        for i in range(len(bounding)):

            # 多个物体相互遮挡的状态（Occluded）
            # kcf_num_blob[i] >= 2 就表明一个 COR 对应多个 TO，也就是多个物体在当前帧中发生了互相遮挡
            if kcf_num_blob[i] >= 2:
                label = -1
                max = 0
                # within_label 中存储的是和当前帧中第 i 个目标框 COR 相匹配的多个 kcf tracker 的目标框 bounding box
                within_label = []

                for j in range(len(rect_in_frame)):
                    if kcf_match[j] == i:
                        within_label.append(j)

                for j in range(len(within_label)):
                    # within_label[j] 这个 kcf tracker 在当前帧中找到了第 i 个匹配
                    # 因此将它的 find_tracker 设置为 i，同时将它的 delay 设置为 0，表明没有失配
                    find_tracker[within_label[j]] = i
                    delay_to_delete[within_label[j]] = 0
                    save = overlap_iou(bounding[i], rect_in_frame[within_label[j]])

                    if save > max:
                        max = save
                        label = within_label[j]

                    # when objects occlude, we group them by saving this unidentified bounding box.
                    x, y, w, h = group_when_occlusion[within_label[j]]
                    if w * h == 0:
                        group_when_occlusion[within_label[j]] = bounding[i]
                    else:
                        area = w * h
                        new_area = bounding[i][2] * bounding[i][3]

                        # update objects' group rectangle
                        # 如果 new_area 相比于 area 的变化不是特别大，那么就使用 new_area 来更新旧的 area
                        if (new_area > area * 0.8) and (new_area < area * 2.2):
                            group_when_occlusion[within_label[j]] = bounding[i]

                    _, boundingbox = tracker_vector[within_label[j]].update(frame)
                    rect_in_frame[within_label[j]] = boundingbox

                    # 判断 within_label[j] 这个 kcf tracker 的信息（即出现的帧数以及路径）是否被保存了
                    if add_this_frame[within_label[j]] == 0:
                        add_this_frame[within_label[j]] = 1
                        # save frame
                        # 保存这个 kcf tracker 跟踪的物体出现的帧数到 rect_frame_save 中
                        rect_frame_save[within_label[j]].append(current_frame)
                        # save rect
                        # 保存这个 kcf tracker 跟踪的物体的位置和大小到 rect_bound_save 中
                        rect_bound_save[within_label[j]].append(rect_in_frame[within_label[j]])
                        cv2.rectangle(frame, rect_in_frame[within_label[j]], (255, 0, 0), 2, 1)

            elif kcf_num_blob[i] == 1:
                label = -1
                for j in range(len(rect_in_frame)):
                    if kcf_match[j] == i:
                        label = j

                find_tracker[label] = i
                delay_to_delete[label] = 0

                area_new = bounding[i][2] * bounding[i][3]
                area_previous = rect_in_frame[label][2] * rect_in_frame[label][3]

                if (area_previous >= 1.4 * area_new) and (area_previous <= 1.8 * area_new):
                    _, rect_in_frame[label] = tracker_vector[label].update(frame)
                else:
                    tracker_vector.pop(label)
                    tracker = cv2.TrackerKCF_create()
                    tracker_vector.append(tracker)
                    tracker.init(frame, bounding[i])
                    rect_in_frame[label] = bounding[i]
                    _, rect_in_frame[label] = tracker.update(frame)

                if add_this_frame[label] == 0:
                    add_this_frame[label] = 1
                    rect_frame_save.append(current_frame)
                    rect_bound_save.append(rect_in_frame[label])
                    cv2.rectangle(frame, rect_in_frame[label], (255, 0, 0), 2, 1)

            else:
                judge = 0
                label = 0

                for m in range(len(rect_in_frame)):
                    tracker_overlap = []

                    if group_when_occlusion[m][2] * group_when_occlusion[m][3] != 0:
                        if overlap_iou(group_when_occlusion[m], bounding[i]) > 0.2:
                            tracker_overlap.append(m)

                            for k in range(len(rect_in_frame)):
                                if k != m:
                                    if group_when_occlusion[k][2] * group_when_occlusion[k][3] != 0:
                                        if overlap_iou(group_when_occlusion[m], group_when_occlusion[k]) > 0.9:
                                            tracker_overlap.append(k)

                            if len(tracker_overlap) == 2:
                                bound_find = 0
                                max = 0

                                for k in range(len(bounding)):
                                    if k != i:
                                        judge = overlap_iou(bounding[k],
                                                            rect_in_frame[tracker_overlap[0]]) + overlap_iou(
                                            bounding[k], rect_in_frame[tracker_overlap[1]])
                                        if judge > max:
                                            max = judge
                                            bound_find = k

                                    if max > 0:
                                        a = bounding[bound_find]
                                        b = rect_in_frame[tracker_overlap[0]]
                                        c = rect_in_frame[tracker_overlap[1]]
                                        if overlap_iou(a, b) > overlap_iou(a, c):
                                            deliver_tracker(tracker_overlap[1], bounding[i])
                                        else:
                                            deliver_tracker(tracker_overlap[0], bounding[i])

                                        judge = 1

                if judge == 0:
                    for k in range(len(rect_in_frame)):
                        if overlap_iou(bounding[i], rect_in_frame[k]) > 0.2:
                            judge = 1
                            break

                if judge == 0:
                    create_new_object(frame, bounding[i])
                    find_tracker.append(i)
                    add_this_frame.append(0)
                    kcf_match.append(0)

        for i in range(len(rect_in_frame)):
            if find_tracker[i] == -1 or current_frame - rect_frame_save[i][len(rect_frame_save[i]) - 1] > 1:
                delay_to_delete[i] += 1

        for i in range(len(rect_in_frame)):
            if delay_to_delete[i] >= 8:
                delete_obj(find_tracker, add_this_frame, i)

    for i in range(len(rect_in_frame)):
        if delay_to_delete[i] != 0:
            cv2.rectangle(frame, rect_in_frame[i], (255, 0, 0), 2, 1)

    print("current frame = " + str(current_frame))


def fetch_video():
    global current_frame

    # 从 filename 读取视频流
    capture = cv2.VideoCapture(filename)
    capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

    if not capture.isOpened():
        print('load video fails')
        return -1

    # calculate whole numbers of frames.
    # 计算视频总共的帧数
    total_frame_num = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    print('total= ' + str(total_frame_num) + ' frames')
    # 设定好视频开始和结束的帧数
    frame_to_stop = total_frame_num
    print('start from ' + str(frame_to_start) + ' frame')

    if frame_to_stop < frame_to_start:
        print('err, no much frame')
    else:
        print('end frame is ' + frame_to_stop)

    rate = capture.get(cv2.CAP_PROP_FPS)
    delay = 1000 / rate
    stop = False

    while not stop:
        ret, frame = capture.read()

        if not ret:
            print('cannot read video.')
            break

        # filepath 表示的是混合高斯背景模型处理过的图片路径
        filepath = 'F:/pycharm/py-projects/learning/mkcf/bgs/rouen_bgs/%08d.png'.format(current_frame)
        # 读取对应视频流的背景模型图片，即 background subtraction
        foreground = cv2.imread(filepath)

        # findContours：函数用来检测物体轮廓
        # countours：是一个双重向量，向量内每个元素保存了一组由连续的 Point 构成的点的集合的向量，每一组点集就是一个轮廓，
        # 有多少轮廓， contours 就有多少元素，并且 cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE 表示只检测最外层轮廓，并且保存轮廓上所有点
        cnts, _ = cv2.findContours(foreground.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # 矩形列表，用来保存前面获取到的轮廓最小矩形
        # boundingRect 用来计算轮廓的垂直边界最小矩形，矩形是与图像上下边界平行的
        bounds = [cv2.boundingRect(c) for c in cnts]

        # 筛除掉不符合条件的目标矩形
        for index, bound in enumerate(bounds):
            # 如果满足以下三个条件中的一个的话：
            # 1.width / height > 6.7
            # 2.width / height < 0.15
            # 3.height * width < blob_size
            # 就把这个目标矩形删除掉，不属于目标
            x, y, w, h = bound
            if (w / h > 6.7) or (w / h < 0.15) or (h * w < min_blob_size):
                bounds.pop(index)

        flag = [0 for _ in range(len(bounds))]
        # 求出所有目标矩形区域的 gravity center
        centroid = [aoi_gravity_center(foreground, bounds[_]) for _ in range(len(bounds))]

        # flag 数组用来表示每一个目标是否应该被删除
        # 检查所有目标矩形框之间的距离，如果距离太小，就将这两个矩形合并成一个更大的矩形
        for i in range(len(bounds)):
            if flag[i] == 1:
                continue

            if bounds[i][2] * bounds[i][3] == 0:
                flag[i] = 1
                continue

            for j in range(i + 1, len(bounds)):
                # 判断两个目标矩形框的中心距离是否小于阈值，如果是的话，就将这两个矩形合并成一个更大的矩形
                # 并且保留其中的 i，而删除掉 j
                if centroid_close_enough(centroid[i], centroid[j]):
                    xi, yi, wi, hi = centroid[i]
                    xj, yj, wj, hj = centroid[j]

                    x = min(xi, xj)
                    y = min(yi, yj)
                    w = max(xi + wi, xj + wj) - x
                    h = max(yi + hi, yj + hj) - y

                    bounds[i] = x, y, w, h
                    # bounds[j] is going to be deleted
                    # 标记 bounds[j] 要被删除
                    flag[j] = 1

        for i in range(len(bounds)):
            if flag[i] == 1:
                bounds.pop(i)
                centroid.pop(i)
                flag.pop(i)

        # 使用多个 KCF 滤波器进行多目标跟踪
        multiple_kcf_tracker(frame, bounds, centroid)
        # 在所有的 kcf tracker 的跟踪框中左上角显示出这个跟踪器的 id
        for i in range(len(rect_in_frame)):
            x, y = rect_in_frame[i]
            cv2.putText(frame, show_msg[i], (x + 6, y + 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # 在画面中显示出所有的 BGS 目标框
        for i in range(len(bounds)):
            x, y, w, h = bounds[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 255, 0), 2, 8, 0)

        cv2.imshow("Original video", frame)
        cv2.imshow("foreground", foreground)

        c = cv2.waitKey(delay)

        if c == 27 or current_frame >= frame_to_stop:
            stop = True

        if c >= 0:
            cv2.waitKey(0)

        current_frame += 1


if __name__ == '__main__':
    fetch_video()
