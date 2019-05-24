# coding=utf-8
from net.yolo3_net import model
from util.load_weights import load_weight
import tensorflow as tf
import sys
import time
import numpy as np


def convert(is_tiny=False):
    if is_tiny:
        anchors = np.array([[1, 1]] * 6)
    else:
        anchors = np.array([[1, 1]] * 9)

    input_data = tf.placeholder(dtype=tf.float32, shape=(1, 416, 416, 3))
    with tf.variable_scope("yolo3"):
        pred = model(input_data, 80, anchors, 'cnn', False, 0.3)

    model_vars_ = tf.global_variables()
    weight_path = sys.argv[1]
    save_path = sys.argv[2]
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
    convert(False)
