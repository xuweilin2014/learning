import numpy as np

def iou(bb_test, bb_gt):
    """
    Computes IoU between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    # IoU = (bb_test 和 bb_gt 框相交部分的面积） / （bb_test 框面积 + bb_gt 框面积 - 两者相交部分的面积）
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o

print(iou(np.array([538,81,580,116]), np.array([553,91,586,129])))
print(iou(np.array([538,79,584,112]), np.array([553,91,586,129])))
print(iou(np.array([545,78,587,110]), np.array([570,110,604,148])))