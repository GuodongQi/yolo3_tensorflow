import time
from collections import defaultdict
from os.path import join, split

import cv2
import numpy as np
import tensorflow as tf

from config.pred_config import get_config
from net.yolo3_net import model
from util.box_utils import pick_box
from util.image_utils import get_color_table, read_image_and_lable
from util.utils import cal_fp_fn_tp_tn, cal_mAP


class YOLO():
    def __init__(self, config):
        self.config = config

        net_type, tiny = split(self.config.weight_path)[-1].split('_')[:2]

        if tiny == 'tiny':
            self.anchor_path = join('model_data', 'yolo_anchors_tiny.txt')
        else:
            self.anchor_path = join('model_data', 'yolo_anchors.txt')

        self.classes = self._get_classes()
        self.anchors = self._get_anchors()
        self.hw = [416, 416]
        self.batch_size = 64
        self.ious_thres = [0.5, 0.75]

        self.test_path = "model_data/test.txt"

        with open(self.test_path) as f:
            self.test_data = f.readlines()

        if tiny == 'tiny':
            assert 6 == len(
                self.anchors), 'the model type does not match with anchors, check anchors or type param'
        else:
            assert 9 == len(
                self.anchors), 'the model type does not match with anchors, check anchors or type param'

        self.input = tf.placeholder(tf.float32, [self.batch_size] + self.hw + [3])
        self.is_training = tf.placeholder(tf.bool, shape=[])
        self.pred = model(self.input, len(self.classes), self.anchors, net_type, self.is_training, False)

        print('start load net_type: {}_{}_model'.format(net_type, tiny))

        # load weights
        conf = tf.ConfigProto()
        conf.gpu_options.allow_growth = True

        # change fraction according to your GPU
        # conf.gpu_options.per_process_gpu_memory_fraction = 0.05

        self.sess = tf.Session(config=conf)
        saver = tf.train.Saver()
        saver.restore(self.sess, self.config.weight_path)
        self.color_table = get_color_table(len(self.classes))

    def _get_anchors(self):
        """loads the anchors from a file"""
        with open(self.anchor_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def _get_classes(self):
        """loads the classes"""
        with open(self.config.classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def test(self):
        total_test_case = len(self.test_data)

        FP_TP = defaultdict(lambda: defaultdict(list))
        GT_NUMS = defaultdict(int)
        GTS = defaultdict(lambda: defaultdict(list))
        DETECTION = defaultdict(lambda: defaultdict(list))
        img_data = []

        print("total test case:", total_test_case)

        for i in range(total_test_case):

            img, xyxy = read_image_and_lable(self.test_data[i], self.hw, is_training=False)
            img_data.append(img)
            print("{}/{}".format(i, total_test_case))
            for per_xyxy in xyxy:
                GTS[i % self.batch_size][self.classes[int(per_xyxy[4])]].append(per_xyxy[:4].tolist())

            if (i + 1) % self.batch_size == 0:  # a batch
                boxes = self.sess.run(self.pred, feed_dict={self.input: img_data, self.is_training: False})

                for b in range(self.batch_size):
                    picked_boxes = pick_box(boxes[b], 0.01, 0.5, self.hw, self.classes)  # NMS
                    for picked_box in picked_boxes:
                        DETECTION[b][self.classes[int(picked_box[5])]].append(picked_box[:5].tolist())

                # cal FP TP
                cal_fp_fn_tp_tn(DETECTION, GTS, FP_TP, GT_NUMS, self.classes, self.ious_thres)

                DETECTION.clear()
                GTS.clear()
                img_data.clear()

        APs, mAPs = cal_mAP(FP_TP, GT_NUMS, self.classes, self.ious_thres)
        print(APs, mAPs)


if __name__ == '__main__':
    configs = get_config()
    YOLO(configs).test()
