import numpy as np
import tensorflow as tf


def box_anchor_iou(b1, b2):
    '''Return iou tensor
    Parameters
    ----------
    b1: tensor, shape=(batch,... 2), wh
    b2: tensor, shape=(j, 2), wh
    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)
    '''

    # Expand dim to apply broadcasting.
    b1 = np.expand_dims(b1, -2)
    b1_mins = - b1 / 2
    b1_maxes = b1 / 2

    # Expand dim to apply broadcasting.
    b2 = np.expand_dims(b2, 0)
    b2_mins = -b2 / 2
    b2_maxes = b2 / 2

    intersect_mins = np.maximum(b1_mins, b2_mins)
    intersect_maxes = np.minimum(b1_maxes, b2_maxes)
    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1[..., 0] * b1[..., 1]
    b2_area = b2[..., 0] * b2[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def pick_box(boxes, score_threshold, hw, classes):
    """
    :param boxes: (boxes_num, 5+numclass),xywh
    :param score_threshold: score_threshold
    :param hw: sacled_image height and width
    :param classes: classes num
    :return:
    """
    score = boxes[..., 4:5] * boxes[..., 5:]
    idx = np.where(score > score_threshold)
    box_select = boxes[idx[:2]]
    box_xywh = box_select[:, :4]
    box_xyxy = wh2xy_np(box_xywh)
    if not len(box_xyxy):
        return []
    box_truncated = []
    for box_k in box_xyxy:
        box_k[0] = box_k[0] if box_k[0] >= 0 else 0
        box_k[1] = box_k[1] if box_k[1] >= 0 else 0
        box_k[2] = box_k[2] if box_k[2] <= hw[1] else hw[1]
        box_k[3] = box_k[3] if box_k[3] <= hw[0] else hw[0]
        box_truncated.append(box_k)
    box_xyxy = np.stack(box_truncated)
    box_socre = score[idx]
    clsid = idx[2]
    picked_boxes = nms_np(
        np.concatenate([box_xyxy, box_socre.reshape([-1, 1]), clsid.reshape([-1, 1])], -1),
        len(classes))
    return picked_boxes


def nms_np(boxes, classes, iou_threshold=0.3, max_output=20):
    """Return nms
    Parameters
    ----------
    :param boxes:  shape=(boxnum 6), xyxy,score,cls
    :param iou_threshold:  iou_threshold
    :param max_output:  max_output
    :param classes:  total_classes_num

    Returns
    -------
    nms boxes
    """

    picked_boxes = []

    for c in range(classes):
        b = boxes[boxes[..., -1] == c]
        score = b[..., 4]
        order = np.argsort(score)
        count = 0
        while order.size > 0 and count < max_output:
            # The index of largest confidence score
            index = order[-1]

            # Pick the bounding box with largest confidence score
            picked_boxes.append(b[index])

            b1_mins = b[index][0:2]
            b1_maxes = b[index][2:4]
            b1_wh = b1_maxes - b1_mins

            b2_mins = b[order[:-1]][..., 0:2]
            b2_maxes = b[order[:-1]][..., 2:4]
            b2_wh = b2_maxes - b2_mins

            intersect_mins = np.maximum(b1_mins, b2_mins)
            intersect_maxes = np.minimum(b1_maxes, b2_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
            b1_area = b1_wh[..., 0] * b1_wh[..., 1]
            b2_area = b2_wh[..., 0] * b2_wh[..., 1]
            iou = intersect_area / (b1_area + b2_area - intersect_area)

            left = np.where(iou < iou_threshold)
            order = order[left]
            count += 1

    return picked_boxes


def xy2wh_np(b):
    """
    :param b:  list xmin ymin xmax ymax
    :return: shape=(...,4) x0 y0 w h
    """
    xmin, ymin, xmax, ymax = b[..., 0:1], b[..., 1:2], b[..., 2:3], b[..., 3:4]
    x0 = (xmin + xmax) / 2.0
    y0 = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    return np.concatenate([x0, y0, w, h], -1)


def wh2xy_np(b):
    """
    :param b: shape=(...,4) x0 y0 w h
    :return: shape=(...,4) xmin ymin xmax ymax
    """
    x0, y0, w, h = b[..., 0:1], b[..., 1:2], b[..., 2:3], b[..., 3:4]
    xmin = x0 - w / 2.0
    xmax = x0 + w / 2.0
    ymin = y0 - h / 2.0
    ymax = y0 + h / 2.0
    return np.concatenate([xmin, ymin, xmax, ymax], -1)


def box_iou(b1, b2):
    '''Return iou tensor
    Parameters
    ----------
    b1: tensor, shape=(batch,... 4), xywh
    b2: tensor, shape=(j, 4), xywh
    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)
    '''

    # Expand dim to apply broadcasting.
    b1 = tf.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = tf.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = tf.maximum(b1_mins, b2_mins)
    intersect_maxes = tf.minimum(b1_maxes, b2_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = tf.math.divide(intersect_area, b1_area + b2_area - intersect_area, name='iou')

    return iou


def xy2wh(b):
    """
    :param b: shape=(...,4) xmin ymin xmax ymax
    :return: shape=(...,4) x0 y0 w h
    """
    xmin, ymin, xmax, ymax = b[..., 0:1], b[..., 1:2], b[..., 2:3], b[..., 3:4]
    x0 = (xmin + xmax) / 2.0
    y0 = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    return tf.concat([x0, y0, w, h], -1)


def wh2xy(b):
    """
    :param b: shape=(...,4) x0 y0 w h
    :return: shape=(...,4) xmin ymin xmax ymax
    """
    x0, y0, w, h = b[..., 0:1], b[..., 1:2], b[..., 2:3], b[..., 3:4]
    xmin = x0 - w / 2.0
    xmax = x0 + w / 2.0
    ymin = y0 - h / 2.0
    ymax = y0 + h / 2.0
    return tf.concat([xmin, ymin, xmax, ymax], -1)


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    b = tf.placeholder(tf.float32, [2, 4, 4])
    xy2wh(b)
