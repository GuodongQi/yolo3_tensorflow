# coding=utf-8
from net.yolo3_net import model
from util.load_weights import load_weight
import tensorflow as tf
import time
import numpy as np
from os.path import join, exists, split
from os import makedirs
import sys


def convert(is_tiny=False):
    if is_tiny:
        anchors = np.array([[1, 1]] * 6)
        weight_path = join('model_data', 'yolov3-tiny.weights')
        save_path = join('logs', 'cnn_tiny', 'cnn_tiny_model')
    else:
        anchors = np.array([[1, 1]] * 9)
        weight_path = join('model_data', 'yolov3.weights')
        save_path = join('logs', 'cnn_full', 'cnn_full_model')

    if not exists(split(save_path)[0]):
        makedirs(split(save_path)[0])
    input_data = tf.placeholder(dtype=tf.float32, shape=(1, 416, 416, 3))

    model(input_data, 80, anchors, 'cnn', False)

    model_vars_ = tf.global_variables()
    assert weight_path.endswith('.weights'), '{} is not a .weights files'.format(weight_path)
    assign_ops_ = load_weight(model_vars_, weight_path)
    t0 = time.time()
    print("start loading weights")
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(assign_ops_)
        saver.save(sess, save_path, write_meta_graph=False, write_state=False)
        t1 = time.time()
        print("convert weights is over, cost {0:.4f}s".format(t1 - t0))


if __name__ == '__main__':
    boolen = sys.argv[1]
    if boolen.lower() == 'tiny':
        convert(True)
    elif boolen.lower() == 'full':
        convert(False)
    else:
        raise Exception('unkonwm argument')
