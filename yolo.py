import time
from os.path import join, split

import cv2
import numpy as np
import tensorflow as tf

from config.pred_config import get_config
from net.yolo3_net import model
from util.box_utils import pick_box, get_true_box
from util.image_utils import get_color_table, plot_img


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
        self.batch_size = 1

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
        # conf.gpu_options.allow_growth = True

        # change fraction according to your GPU
        conf.gpu_options.per_process_gpu_memory_fraction = 0.05
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

    def forward(self, img):
        """
        :param img:  shape = (h,w,c), 0-255
        :return:
        """
        height, width = img.shape[:2]
        img_ = cv2.resize(img, tuple(self.hw)[::-1])
        h_r = height / self.hw[0]
        w_r = width / self.hw[1]

        im_data = np.expand_dims(img_[..., ::-1], 0) / 255.0
        boxes = self.sess.run(self.pred, feed_dict={self.input: im_data, self.is_training: False})

        vis_img = []
        for b in range(self.batch_size):
            picked_boxes = pick_box(boxes[b], 0.3, 0.6, self.hw, self.classes)
            true_boxes = get_true_box(picked_boxes, w_r, h_r)
            per_img = img
            per_img = plot_img(per_img, true_boxes, self.color_table, self.classes)
            print('find {} boxes'.format(len(true_boxes)))
            print(true_boxes)
            vis_img.append(per_img)
        return vis_img[0]

    def detect_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return None
        img = self.forward(img)
        cv2.imshow('img', img)
        cv2.imwrite('tiny.jpg', img)
        cv2.waitKey(0)
        return 1

    def detect_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("Couldn't open webcam or video")
        # video_FourCC = -1
        video_FourCC = cv2.VideoWriter_fourcc(*'XVID')
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # writer = cv2.VideoWriter('output.mp4', video_FourCC, video_fps, (width, height))
        writer = cv2.VideoWriter('output.avi', video_FourCC, video_fps, (width, height))

        total_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        time1 = time.time()

        while True:
            ret, frame = cap.read()
            if ret:
                out = self.forward(frame)
                time2 = time.time()
                d_time = time2 - time1
                time1 = time2
                total_time += d_time
                curr_fps += 1
                if total_time >= 1:
                    fps = "FPS: {}".format(curr_fps)
                    total_time -= 1
                    curr_fps = 0

                out = cv2.putText(out, fps, tuple(np.int32([20, 30])),
                                  cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255))
                out = cv2.resize(out, (width, height))
                cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
                cv2.imshow('result', out)
                cv2.waitKey(1)
                writer.write(out)
            else:
                break


if __name__ == '__main__':
    configs = get_config()
    yolo = YOLO(configs)
    if configs.video:
        yolo.detect_video(configs.video)
    elif configs.image:
        yolo.detect_image(configs.image)
    else:
        while True:
            img_path = input('input image path:')
            if not yolo.detect_image(img_path):
                print('check your iamge path ')
            continue
