import tensorflow as tf

from util.box_utils import box_iou, xy2wh, wh2xy

"""
(1280 * 640)
input = (640 * 320)
640 * 320
320 * 160
160 * 80
80 * 40
40 * 20
20 * 10
10 * 5
"""
leaky_alpha = 0.1

xavier_initializer = tf.initializers.glorot_uniform()


def conv_block(x, filters, stride, out_channel, net_type, name='', relu=True):
    """
    :param x: input :nhwc
    :param filters: list [f_w, f_h]
    :param stride: list int
    :param out_channel: int, out_channel
    :param net_type: cnn mobilenet
    :param name: str
    :param relu: boolean
    :return: depwise and pointwise out
    """
    with tf.name_scope('' + name):
        in_channel = x.shape[3].value
        if net_type == 'cnn':
            with tf.name_scope('cnn'):
                # weight = tf.Variable(tf.truncated_normal([filters[0], filters[1], in_channel, out_channel], 0, 0.01))
                weight = tf.Variable(xavier_initializer([filters[0], filters[1], in_channel, out_channel]))
                x = tf.nn.conv2d(x, weight, [1, stride[0], stride[1], 1], 'SAME')
                if relu:
                    x = tf.layers.batch_normalization(x)
                    x = tf.nn.leaky_relu(x, leaky_alpha)
                else:
                    bias = tf.Variable(tf.zeros_like(x[0]))
                    x += bias
        elif net_type == 'mobilenetv1':
            with tf.name_scope('depthwise'):
                # depthwise_weight = tf.Variable(tf.truncated_normal([filters[0], filters[1], in_channel, 1], 0, 0.01))
                depthwise_weight = tf.Variable(xavier_initializer([filters[0], filters[1], in_channel, 1]))
                x = tf.nn.depthwise_conv2d(x, depthwise_weight, [1, stride[0], stride[1], 1], 'SAME')
                x = tf.layers.batch_normalization(x)
                x = tf.nn.relu6(x)

            with tf.name_scope('pointwise'):
                # pointwise_weight = tf.Variable(tf.truncated_normal([1, 1, in_channel, out_channel], 0, 0.01))
                pointwise_weight = tf.Variable(xavier_initializer([1, 1, in_channel, out_channel]))
                x = tf.nn.conv2d(x, pointwise_weight, [1, 1, 1, 1], 'SAME')
                if relu:
                    x = tf.layers.batch_normalization(x)
                    x = tf.nn.relu6(x)
                else:
                    bias = tf.Variable(tf.zeros_like(x[0]))
                    x += bias

        elif net_type == 'mobilenetv2':
            tmp_channel = out_channel * 3
            with tf.name_scope('expand_pointwise'):
                pointwise_weight = tf.Variable(xavier_initializer([1, 1, in_channel, tmp_channel]))
                x = tf.nn.conv2d(x, pointwise_weight, [1, 1, 1, 1], 'SAME')
                x = tf.layers.batch_normalization(x)
                x = tf.nn.relu6(x)
            with tf.name_scope('depthwise'):
                depthwise_weight = tf.Variable(xavier_initializer([filters[0], filters[1], tmp_channel, 1]))
                x = tf.nn.depthwise_conv2d(x, depthwise_weight, [1, stride[0], stride[1], 1], 'SAME')
            with tf.name_scope('project_pointwise'):
                pointwise_weight = tf.Variable(xavier_initializer([1, 1, tmp_channel, out_channel]))
                x = tf.nn.conv2d(x, pointwise_weight, [1, 1, 1, 1], 'SAME')
                if relu:
                    x = tf.layers.batch_normalization(x)
                else:
                    bias = tf.Variable(tf.zeros_like(x[0]))
                    x += bias
        else:
            raise Exception('net type is error, please check')
    return x


def residual(x, net_type, out_channel=1, expand_time=1, stride=1):
    if net_type in ['cnn', 'mobilenetv1']:
        out_channel = x.shape[3].value
        shortcut = x
        x = conv_block(x, [1, 1], [1, 1], out_channel // 2, net_type='cnn')
        x = conv_block(x, [3, 3], [1, 1], out_channel, net_type='cnn')
        x += shortcut

    elif net_type == 'mobilenetv2':
        shortcut = x
        in_channel = x.shape[3].value
        tmp_channel = in_channel * expand_time
        with tf.name_scope('expand_pointwise'):
            pointwise_weight = tf.Variable(xavier_initializer([1, 1, in_channel, tmp_channel]))
            x = tf.nn.conv2d(x, pointwise_weight, [1, 1, 1, 1], 'SAME')
            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu6(x)
        with tf.name_scope('depthwise'):
            depthwise_weight = tf.Variable(xavier_initializer([3, 3, tmp_channel, 1]))
            x = tf.nn.depthwise_conv2d(x, depthwise_weight, [1, stride, stride, 1], 'SAME')
        with tf.name_scope('project_pointwise'):
            pointwise_weight = tf.Variable(xavier_initializer([1, 1, tmp_channel, out_channel]))
            x = tf.nn.conv2d(x, pointwise_weight, [1, 1, 1, 1], 'SAME')
            x = tf.layers.batch_normalization(x)
        x += shortcut

    return x


def full_yolo_body(x, net_type):
    channel = x.shape[-1].value
    if net_type in ['mobilenetv2']:
        net_type = 'mobilenetv1'
    x = conv_block(x, [1, 1], [1, 1], channel // 2, net_type)
    x = conv_block(x, [3, 3], [1, 1], channel, net_type)
    x = conv_block(x, [1, 1], [1, 1], channel // 2, net_type)
    x = conv_block(x, [3, 3], [1, 1], channel, net_type)
    x_route = x
    x = conv_block(x, [1, 1], [1, 1], channel // 2, net_type)
    x = conv_block(x, [3, 3], [1, 1], channel, net_type)
    return x_route, x


def full_darknet_body(x, net_type):
    """
    yolo3_tiny build by net_type
    :param x:
    :param net_type: cnn mobilenet
    :return:
    """
    if net_type in ['cnn', 'mobilenetv1']:
        x = conv_block(x, [3, 3], [1, 1], 32, 'cnn')

        # down sample
        x = conv_block(x, [3, 3], [2, 2], 64, 'cnn')
        for i in range(1):
            x = residual(x, net_type)

        # down sample
        x = conv_block(x, [3, 3], [2, 2], 128, 'cnn')
        for i in range(2):
            x = residual(x, net_type)

        # down sample
        x = conv_block(x, [3, 3], [2, 2], 256, 'cnn')
        for i in range(8):
            x = residual(x, net_type)
        route2 = x

        # down sample
        x = conv_block(x, [3, 3], [2, 2], 512, 'cnn')
        for i in range(8):
            x = residual(x, net_type)
        route1 = x

        # down sample
        x = conv_block(x, [3, 3], [2, 2], 1024, 'cnn')
        for i in range(4):
            x = residual(x, net_type)

    elif net_type == 'mobilenetv2':
        # down sample
        x = conv_block(x, [3, 3], [2, 2], 32, 'cnn')

        # down sample
        x = conv_block(x, [3, 3], [2, 2], 64, net_type)
        for i in range(2):
            x = residual(x, net_type, 64, 1)

        # down sample
        x = conv_block(x, [3, 3], [2, 2], 96, net_type)
        for i in range(4):
            x = residual(x, net_type, 96, 6)
        route2 = x

        # down sample
        x = conv_block(x, [3, 3], [2, 2], 160, net_type)
        for i in range(4):
            x = residual(x, net_type, 160, 6)
        route1 = x

        # down sample
        x = conv_block(x, [3, 3], [2, 2], 320, net_type)
        for i in range(3):
            x = residual(x, net_type, 320, 1)

    return x, route1, route2


def full_yolo_head(x, route1, route2, num_class, anchors, net_type):
    with tf.name_scope('body_layer1'):
        x_route, x = full_yolo_body(x, net_type)
    x = conv_block(x, [1, 1], [1, 1], 3 * (5 + num_class), 'cnn', "yolo_head1", False)
    fe1, grid1 = yolo(x, anchors[[0, 1, 2]])

    with tf.name_scope('head_layer2'):
        x = conv_block(x_route, [1, 1], [1, 1], route1.shape[-1].value, net_type)
        transpose_weight = tf.Variable(xavier_initializer([1, 1, route1.shape[-1].value, route1.shape[-1].value]))
        x = tf.nn.conv2d_transpose(x, transpose_weight,
                                   [x.shape[0].value, x.shape[1].value * 2, x.shape[2].value * 2, x.shape[3].value],
                                   [1, 2, 2, 1], 'SAME')
        x = tf.concat([x, route1], 3)
        x_route, x = full_yolo_body(x, net_type)
    x = conv_block(x, [1, 1], [1, 1], 3 * (5 + num_class), 'cnn', "yolo_head2", False)
    fe2, grid2 = yolo(x, anchors[[3, 4, 5]])

    with tf.name_scope('head_layer3'):
        x = conv_block(x_route, [1, 1], [1, 1], route2.shape[-1].value, net_type)
        transpose_weight = tf.Variable(xavier_initializer([1, 1, route2.shape[-1].value, route2.shape[-1].value]))
        x = tf.nn.conv2d_transpose(x, transpose_weight,
                                   [x.shape[0].value, x.shape[1].value * 2, x.shape[2].value * 2, x.shape[3].value],
                                   [1, 2, 2, 1], 'SAME')
        x = tf.concat([x, route2], 3)
        x_route, x = full_yolo_body(x, net_type)
    x = conv_block(x, [1, 1], [1, 1], 3 * (5 + num_class), 'cnn', "yolo_head3", False)
    fe3, grid3 = yolo(x, anchors[[6, 7, 8]])

    fe = tf.concat([fe1, fe2, fe3], 1)
    return fe, grid1, grid2, grid3


def tiny_darknet_body(x, net_type):
    """
    yolo3_tiny build by net_type
    :param x:
    :param net_type: cnn or mobile-net
    :return:
    """
    if net_type in ['mobilenetv1', 'mobilenetv2']:
        net_type = 'mobilenetv1'
    x = conv_block(x, [3, 3], [1, 1], 16, net_type)
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    x = conv_block(x, [3, 3], [1, 1], 32, net_type)
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    x = conv_block(x, [3, 3], [1, 1], 64, net_type)
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    x = conv_block(x, [3, 3], [1, 1], 128, net_type)
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    x_route = x

    x = conv_block(x, [3, 3], [1, 1], 256, net_type)
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    x = conv_block(x, [3, 3], [1, 1], 512, net_type)
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 1, 1, 1], 'SAME')

    x = conv_block(x, [3, 3], [1, 1], 1024, net_type)

    return x, x_route


def tiny_yolo_head(x, x_route1, num_class, anchors, net_type):
    with tf.name_scope('head_layer1'):
        x = conv_block(x, [1, 1], [1, 1], 256, net_type)
        x_route2 = x
        x = conv_block(x, [3, 3], [1, 1], 512, net_type)
        x = conv_block(x, [1, 1], [1, 1], 3 * (5 + num_class), 'cnn', "yolo_head1")
        fe1 = x
        fe1, grid1 = yolo(fe1, anchors[[0, 1, 2]])

    with tf.name_scope('head_layer2'):
        x = conv_block(x_route2, [1, 1], [1, 1], 128, net_type)
        transpose_weight = tf.Variable(xavier_initializer([1, 1, 128, 128]))
        x = tf.nn.conv2d_transpose(x, transpose_weight,
                                   [x.shape[0].value, x.shape[1].value * 2, x.shape[2].value * 2, x.shape[3].value],
                                   [1, 2, 2, 1], 'SAME')
        x = tf.concat([x, x_route1], 3)
        x = conv_block(x, [3, 3], [1, 1], 256, net_type)
        x = conv_block(x, [1, 1], [1, 1], 3 * (5 + num_class), 'cnn', "yolo_head2")
        fe2 = x
        fe2, grid2 = yolo(fe2, anchors[[3, 4, 5]])

    fe = tf.concat([fe1, fe2], 1)
    return fe, grid1, grid2


def yolo(f, anchors):
    """
    convert feature to box and scores
    :param f:
    :param anchors:
    :return:
    """
    anchor_tensor = tf.constant(anchors, tf.float32)
    batchsize = f.shape[0]
    f = tf.reshape(f, [f.shape[0], f.shape[1], f.shape[2], 3, -1])
    grid_y = tf.tile(tf.reshape(tf.range(f.shape[1]), [1, -1, 1, 1]), [batchsize, 1, f.shape[2], 1])
    grid_x = tf.tile(tf.reshape(tf.range(f.shape[2]), [1, 1, -1, 1]), [batchsize, f.shape[1], 1, 1])
    grid = tf.tile(tf.cast(tf.concat([grid_x, grid_y], -1), tf.float32)[:, :, :, tf.newaxis, :], (1, 1, 1, 3, 1))

    box_xy = (tf.nn.sigmoid(f[..., :2]) + grid) / tf.cast(grid.shape[::-1][2:4], tf.float32, )
    box_wh = tf.math.exp(f[..., 2:4]) * anchor_tensor
    # box_confidence = tf.nn.sigmoid(f[..., 4:5])
    # classes_score = tf.nn.sigmoid(f[..., 5:])
    box_confidence = f[..., 4:5]
    classes_score = f[..., 5:]
    feas = tf.reshape(tf.concat([box_xy, box_wh, box_confidence, classes_score], -1), [batchsize, -1, 3, f.shape[4]])
    return feas, grid


def model(x, num_classes, anchors, net_type, cal_loss=False, score_threshold=0.3):
    batchsize, height, width, _ = x.get_shape().as_list()
    if len(anchors) == 6:
        x, x_route = tiny_darknet_body(x, net_type)
        y, *grid = tiny_yolo_head(x, x_route, num_classes, anchors, net_type)
    else:
        x, route1, route2 = full_darknet_body(x, net_type)
        y, *grid = full_yolo_head(x, route1, route2, num_classes, anchors, net_type)

    box_xy, box_wh, box_confidence, classes_score = y[..., :2], y[..., 2:4], y[..., 4:5], y[..., 5:]
    box_xy *= tf.constant([width, height], tf.float32)
    # box_wh *= tf.constant([width, height], tf.float32)
    boxe = tf.concat([box_xy, box_wh, box_confidence, classes_score], -1, name='debug_pred')

    if cal_loss:
        return boxe, grid
    else:
        return boxe


def loss(pred, gts, anchors, input_size, lambda_coord, lambda_noobj, lambda_cls, iou_threshold, debug_=False):
    """
    :param pred: (batch_size, num_boxes, 3, 5+num_class)[x0 y0 w h ] +grid
    :param gts: shape = (batch_size, num_boxes, 3, 4+num_class) [xywh,calsses]
    :param anchors:
    :param input_size: height * width
    :param lambda_coord: lambda
    :param lambda_noobj: lambda
    :param lambda_cls: lambda
    :param iou_threshold: iou_threshold
    :param debug_:
    :return:
    """

    def binary_cross(labels, pred):
        # pred = tf.clip_by_value(pred, 1e-10, 1 - 1e-10)
        # return -labels * tf.math.log(pred)
        # pred = tf.math.log(pred / (1 - pred))
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=pred)

    pred_boxes, grid = pred

    masks = gts[..., 4]
    batchsize = masks.shape[0].value
    i_height, i_width = input_size

    # cal ignore_mask
    ignore_mask = []
    for b in range(batchsize):
        true_box = tf.boolean_mask(gts[b:b + 1, ..., :4], masks[b:b + 1], name='debug_true_box')
        with tf.name_scope('debug_iou'):
            ious = box_iou(pred_boxes[b:b + 1, ..., :4], true_box)
        ious = tf.reduce_max(ious, -1)
        ignore_mask_ = tf.where(ious > iou_threshold, tf.zeros_like(ious), tf.ones_like(ious))
        ignore_mask.append(ignore_mask_)
    ignore_mask = tf.concat(ignore_mask, 0, name='debug_ignore_mask')

    scale_tensor = []
    grid_tensor = []
    anchors_tnesor = []
    for ii, g in enumerate(grid):
        _, g_h, g_w, g_n, _ = g.get_shape().as_list()
        scale = i_height / g_h
        scale_tensor.append(tf.constant(scale, tf.float32, [batchsize, g_h * g_w, g_n, 2]))
        grid_tensor.append(tf.reshape(g, [batchsize, g_h * g_w, g_n, 2]))
        anchors_ = tf.constant(anchors[[ii, ii + 1, ii + 2]], dtype=tf.float32)
        anchors_tnesor.append(tf.tile(anchors_[tf.newaxis, tf.newaxis, :, :], [batchsize, g_h * g_w, 1, 1]))
    scale_tensor = tf.concat(scale_tensor, 1, name="debug_scale")
    grid_tensor = tf.concat(grid_tensor, 1, name="debug_grid")
    anchors_tnesor = tf.concat(anchors_tnesor, 1, name="debug_anchor_mask")

    raw_pred_xy = tf.math.subtract(tf.math.divide(pred_boxes[..., :2], scale_tensor, name='debug_pred_div_scale'),
                                   grid_tensor,
                                   name='debug_raw_pred_xy')
    raw_gt_xy = tf.math.subtract(gts[..., :2] / scale_tensor,
                                 tf.tile(masks[..., tf.newaxis], [1, 1, 1, 2]) * grid_tensor, name='debug_raw_gts_xy')

    raw_pred_wh = tf.math.log(pred_boxes[..., 2:4] / anchors_tnesor + 1e-15, name='debug_raw_pred_wh')
    raw_gt_wh = tf.math.multiply(tf.tile(masks[..., tf.newaxis], [1, 1, 1, 2]),
                                 tf.math.log(gts[..., 2:4] / anchors_tnesor + 1e-15), name='debug_raw_gt_wh')

    vars = tf.trainable_variables()
    l2_loss = tf.reduce_sum([tf.nn.l2_loss(var) for var in vars]) * 0.001
    # l2_loss = 0

    masks_noobj = (1 - masks) * ignore_mask

    # n_xywh = tf.reduce_sum(masks, name='debug_n_xywh')
    # n_noob = tf.reduce_sum(masks_noobj, name='debug_n_noobj') / 100
    n_xywh = batchsize
    n_noob = batchsize

    loss_xy = tf.reduce_sum(
        lambda_coord * masks * tf.reduce_sum(
            tf.math.square(raw_gt_xy - raw_pred_xy),
            -1), name='debug_loss_xy') / n_xywh
    loss_wh = tf.reduce_sum(
        lambda_coord * masks * tf.reduce_sum(
            tf.math.square(raw_pred_wh - raw_gt_wh),
            -1), name='debug_loss_wh') / n_xywh
    loss_obj_confidence = tf.reduce_sum(
        masks * binary_cross(labels=masks, pred=pred_boxes[..., 4]), name='debug_loss_obj') / n_xywh

    loss_noobj_confidence = tf.reduce_sum(
        lambda_noobj * masks_noobj * binary_cross(labels=masks, pred=pred_boxes[..., 4]),
        name='debug_loss_noobj') / n_noob
    loss_cls = tf.reduce_sum(
        masks * lambda_cls * tf.reduce_sum(
            binary_cross(labels=gts[..., 5:], pred=pred_boxes[..., 5:]), -1), name='debug_loss_cls'
    ) / n_xywh
    if debug_:
        p = tf.print("loss_xy", loss_xy, "loss_wh", loss_wh, "loss_obj_confidence", loss_obj_confidence,
                     'loss_noobj_confidence', loss_noobj_confidence, "loss_cls", loss_cls, "l2_loss", l2_loss)
        with tf.control_dependencies([p]):
            return loss_xy + loss_wh + loss_obj_confidence + loss_noobj_confidence + loss_cls + l2_loss
    return loss_xy + loss_wh + loss_obj_confidence + loss_noobj_confidence + loss_cls + l2_loss
