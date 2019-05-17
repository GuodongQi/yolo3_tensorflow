from os import getcwd
from os.path import join, split
import tensorflow as tf
import numpy as np
import cv2

from net.yolo3_net import model
from util.box_utils import np_sigmoid, wh2xy_np, nms_np
from util.pred_config import get_config


class YOLO():
    def __init__(self):
        self.classes = self._get_classes()
        self.anchors = self._get_anchors()
        self.hw = [320, 640]

        net_type, tiny = split(config.weight_path)[-1].split('_')[:2]
        if tiny == 'tiny':
            assert 6 == len(
                self.anchors), 'the model type does not match with anchors, check anchors or type param'
        else:
            assert 9 == len(
                self.anchors), 'the model type does not match with anchors, check anchors or type param'

        self.input = tf.placeholder(tf.float32, [1] + self.hw + [3])
        self.pred = model(self.input, len(self.classes), self.anchors, net_type, False)

        # load weights
        conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        self.sess = tf.Session(config=conf)
        saver = tf.train.Saver()
        saver.restore(self.sess, config.weight_path)

    def _get_anchors(self):
        """loads the anchors from a file"""
        with open(config.anchor_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def _get_classes(self):
        """loads the classes"""
        with open(config.classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def forward(self, im_data):
        """
        :param im_data:  shape = (h,w,c)
        :return:
        """
        im_data = np.expand_dims(im_data, 0)
        boxes = self.sess.run(self.pred, feed_dict={self.input: im_data})
        score = np_sigmoid(boxes[..., 4:5]) * np_sigmoid(boxes[..., 5:])
        b = 0
        idx = np.where(score[b] > 0.5)
        box_select = boxes[b][idx[:2]]
        box_xywh = box_select[:, :4]
        box_xyxy = wh2xy_np(box_xywh)
        box_socre = score[b][idx]
        clsid = idx[2]
        picked_boxes = nms_np(
            np.concatenate([box_xyxy, box_socre.reshape([-1, 1]), clsid.reshape([-1, 1])], -1),
            len(self.classes))
        per_img = (im_data[b] + 1) * 128
        for bbox in picked_boxes:
            per_img = cv2.rectangle(per_img, tuple(np.int32([bbox[0], bbox[1]])),
                                    tuple(np.int32([bbox[2], bbox[3]])), (0, 255, 0), 2)
            per_img = cv2.putText(per_img, "{} {:.2f}".format(self.classes[int(bbox[5])], bbox[4]),
                                  tuple(np.int32([bbox[0], bbox[1]])),
                                  cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0))
        return per_img

    def detect_image(self, img_path):
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        img = cv2.resize(img, tuple(self.hw)[::-1])[:, :, ::-1] / 255
        out = self.forward(img)
        out = cv2.resize(out, (width, height))
        cv2.imshow('img', out[:, :, ::-1])
        cv2.waitKey()


if __name__ == '__main__':
    config = get_config()
    yolo = YOLO()
    if config.video:
        yolo.detect_image(config.i)
    elif config.image:
        yolo.detect_image(config.i)
    else:

        img_path = input('input image path:')
        yolo.detect_image(img_path)

