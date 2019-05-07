import tensorflow as tf


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


def xy2wh_np(b):
    """
    :param b:  list xmin ymin xmax ymax
    :return: shape=(...,4) x0 y0 w h
    """
    xmin, ymin, xmax, ymax = b
    x0 = (xmin + xmax) / 2.0
    y0 = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    return [x0, y0, w, h]


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


if __name__ == '__main__':
    b = tf.placeholder(tf.float32, [2, 4, 4])
    xy2wh(b)
