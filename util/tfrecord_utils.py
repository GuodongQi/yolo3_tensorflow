import random
import tensorflow as tf
import numpy as np

from util.box_utils import box_anchor_iou_tf


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities.
    Taken from
    https://github.com/tensorflow/models/blob/master/inception/inception/data/build_image_data.py
    """

    def __init__(self):
        # Create a single Session to run all image coding calls.
        conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        self._sess = tf.Session(config=conf)

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(
            image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(
            self._decode_jpeg_data, channels=3)

        self._encode_jpeg_data = tf.placeholder(dtype=tf.uint8)
        self._encode_jpeg = tf.image.encode_jpeg(
            self._encode_jpeg_data, format='rgb')

        self._decode_png_data = tf.placeholder(dtype=tf.string)
        self._decode_png = tf.image.decode_png(
            self._decode_png_data, channels=3)

        self._encode_png_data = tf.placeholder(dtype=tf.uint8)
        self._encode_png = tf.image.encode_png(self._encode_png_data)

    def png_to_jpeg(self, image_data):
        return self._sess.run(
            self._png_to_jpeg, feed_dict={
                self._png_data: image_data
            })

    def decode_jpeg(self, image_data):
        image = self._sess.run(
            self._decode_jpeg, feed_dict={
                self._decode_jpeg_data: image_data
            })
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

    def encode_jpeg(self, image):
        image_data = self._sess.run(
            self._encode_jpeg, feed_dict={
                self._encode_jpeg_data: image
            })
        return image_data

    def encode_png(self, image):
        image_data = self._sess.run(
            self._encode_png, feed_dict={
                self._encode_png_data: image
            })
        return image_data

    def decode_png(self, image_data):
        image = self._sess.run(
            self._decode_png, feed_dict={
                self._decode_png_data: image_data
            })
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list) and not isinstance(value, np.ndarray):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list) and not isinstance(value, np.ndarray):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_example(image_data, image_path, height, width, label):
    """
    Build an Example proto for an image example.
    :param image_data: string, JPEG encoding of RGB image
    :param image_path: string, path to this image file
    :param height: integers, image shapes in pixels.
    :param width: integers, image shapes in pixels.
    :param label: boxes, [xmim, ymin, xmax, ymax, clsid] * num_obj
    """
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/image_data': bytes_feature(tf.compat.as_bytes(image_data)),
        'image/image_path': bytes_feature(tf.compat.as_bytes(image_path)),
        'label': int64_feature(label)
    }))
    return example


def read_tfrecords(files, do_shuffle=True):
    fqueue = tf.train.string_input_producer(files, shuffle=do_shuffle, name='input')
    # fqueue = tf.data.TFRecordDataset(files)
    reader = tf.TFRecordReader()
    _, example_serialized = reader.read(fqueue)
    feature_map = {
        'image/height': tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
        'image/width': tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
        'image/image_data': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/image_path': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'label': tf.VarLenFeature(dtype=tf.int64)
    }

    features = tf.parse_single_example(example_serialized, feature_map)

    height = tf.cast(features['image/height'], dtype=tf.int32)
    width = tf.cast(features['image/width'], dtype=tf.int32)
    label = tf.sparse_tensor_to_dense(features['label'])
    label = tf.reshape(label, [-1, 5])
    label = tf.cast(label, tf.int32)

    image_path = tf.cast(features['image/image_path'], dtype=tf.string)
    image_data = tf.image.decode_jpeg(features['image/image_data'])
    # convert to [0, 1]
    image_data = tf.image.convert_image_dtype(image_data, tf.float32)
    image_size = tf.concat([height, width], 0)
    return image_data, image_path, image_size, label


def rand(a=0., b=1.):
    return random.random() * (b - a) + a


def processing_image(feature):
    image_data, image_path, image_size, label = feature

    if rand() < 0.5:
        image_data = distort_color(image_data, 0, False)

    if rand() < 0.5:
        image_data, label = crop_and_resise_images(image_data, image_size, label)

    if rand() < 0.2:
        image_data = tf.image.flip_left_right(image_data)
        label = tf.concat([
            image_size[1] - label[:, 2:3],
            label[:, 1:2],
            image_size[1] - label[:, 0:1],
            label[:, 3:4],
            label[:, 4:5],
        ], 1)

    if rand() < 0.1:
        image_data = tf.image.flip_up_down(image_data)
        label = tf.concat([
            label[:, 0:1],
            image_size[0] - label[:, 3:4],
            label[:, 2:3],
            image_size[0] - label[:, 1:2],
            label[:, 4:5],
        ], 1)

    return image_data, image_path, image_size, label


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
    """Distort the color of a Tensor image.
    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.
    Args:
        image: 3-D Tensor containing single image in [0, 1].
        color_ordering: Python int, a type of distortion (valid values: 0-3).
        fast_mode: Avoids slower ops (random_hue and random_contrast)
        scope: Optional scope for name_scope.
    Returns:
        3-D Tensor color-distorted image on range [0, 1]
    Raises:
        ValueError: if color_ordering not in [0, 3]
    """
    with tf.name_scope(scope, 'distort_color', [image]):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                raise ValueError('color_ordering must be in [0, 3]')

    return tf.clip_by_value(image, 0.0, 1.0)


def crop_and_resise_images(image_data, image_size, label):
    bounding_boxes_rate = tf.concat([label[:, 0:1] / image_size[1],
                                     label[:, 1:2] / image_size[0],
                                     label[:, 2:3] / image_size[1],
                                     label[:, 3:4] / image_size[0],
                                     ], 1)
    bounding_boxes_rate = tf.cast(tf.expand_dims(bounding_boxes_rate, 0), dtype=tf.float32)

    begin, size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image_data), bounding_boxes_rate,
                                                            min_object_covered=0.5, area_range=[0.5, 1])
    image_data = tf.slice(image_data, begin, size)
    image_data = tf.image.resize(image_data, image_size)

    # correct bounding_boxes
    bxmin = tf.cast((label[:, 0:1] - begin[1]) * image_size[1] / size[1], tf.int32)
    bymin = tf.cast((label[:, 1:2] - begin[0]) * image_size[0] / size[0], tf.int32)
    bxmax = tf.cast((label[:, 2:3] - begin[1]) * image_size[1] / size[1], tf.int32)
    bymax = tf.cast((label[:, 3:4] - begin[0]) * image_size[0] / size[0], tf.int32)

    # correct boxes which locate outside of images
    bxmin = tf.where(bxmin > image_size[1] - 1, (image_size[1] - 1) * tf.ones_like(bxmin), bxmin)
    bymin = tf.where(bymin > image_size[0] - 1, (image_size[0] - 1) * tf.ones_like(bymin), bymin)
    bxmax = tf.where(bxmax > image_size[1] - 1, (image_size[1] - 1) * tf.ones_like(bxmax), bxmax)
    bymax = tf.where(bymax > image_size[0] - 1, (image_size[0] - 1) * tf.ones_like(bymax), bymax)

    bounding_boxes = tf.concat([bxmin, bymin, bxmax, bymax], 1)
    bounding_boxes = tf.where(bounding_boxes < 0, tf.zeros_like(bounding_boxes), bounding_boxes)

    label = tf.concat([bounding_boxes, label[:, 4:5]], 1)

    # delete boxes which locate outside of images
    idx_w = tf.greater(bxmax, bxmin)
    idx_h = tf.greater(bymax, bymin)
    idx = tf.logical_and(idx_w, idx_h)
    idx = tf.reshape(idx, [-1])

    label = tf.boolean_mask(label, idx)

    return image_data, label


def convert_label(label, gds_init, anchor, is_val):
    """
    convert boxes to grid style
    :param label: tensor shape=(?,5), xmin,ymin,xmax,ymax,class
    :param gds_init: tensor, a grid,  associated with input and net type
    :param anchor: tensor, anchor box
    :param is_val: bool, whether for val or not
    :return:
    """
    box_wh = tf.concat([label[:, 2:3] - label[:, 0:1], label[:, 3:4] - label[:, 1:2]], 1)
    box_iou = box_anchor_iou_tf(box_wh, anchor)
    k = tf.argmax(box_iou, -1)

    return


if __name__ == '__main__':
    dataset_dir = '/media/data2/qiguodong/new/tfrecords/train'
    import os

    fs = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir)]

    x = read_tfrecords(fs)

    from net.yolo3_net import model


    def __get_anchors():
        """loads the anchors from a file"""
        with open('/media/data2/qiguodong/new/v2/yolo3_tensorflow/model_data/yolo_anchors.txt') as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)


    def __get_classes():
        """loads the classes"""
        with open('/media/data2/qiguodong/new/v2/yolo3_tensorflow/model_data/coco_classes.txt') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names


    anchors = __get_anchors()
    inputs = tf.placeholder(tf.float32, [4] + [608, 608] + [3])
    is_training = tf.placeholder(tf.bool, shape=[])
    pred = model(inputs, 80, anchors, 'cnn', is_training, True)

    grid_shape = [g.get_shape().as_list() for g in pred[2]]
    gds_init = [np.zeros(g_shape[1:3] + [3, 9 + 80]) for g_shape in grid_shape]

    anchors = tf.convert_to_tensor(anchors)
    convert_label(x[-1], gds_init, anchors, False)
    with tf.Session() as sess:
        coor = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coor)
        import cv2
        import numpy as np

        for i in range(1000):
            f1, f2 = sess.run([x, processing_image(x)])
            imgdata1, *_ = f1
            imgdata, image_path, image_size, label = f2

            print(label)
            for b in label:
                imgdata = cv2.rectangle(imgdata, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0))
            imgdata2 = np.hstack([imgdata1, imgdata])
            cv2.imshow('win', imgdata2[:, :, ::-1])

            cv2.waitKey(0)
