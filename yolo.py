import time
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
        net_type, tiny = split(config.weight_path)[-1].split('_')[:2]

        if tiny == 'tiny':
            self.anchor_path = join('model_data', 'yolo_anchors_tiny.txt')
        else:
            self.anchor_path = join('model_data', 'yolo_anchors.txt')

        self.classes = self._get_classes()
        self.anchors = self._get_anchors()
        self.hw = [416, 416]

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
        with open(self.anchor_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def _get_classes(self):
        """loads the classes"""
        with open(config.classes_path) as f:
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
        boxes = self.sess.run(self.pred, feed_dict={self.input: im_data})
        score = np_sigmoid(boxes[..., 4:5]) * np_sigmoid(boxes[..., 5:])
        b = 0

        # nms
        idx = np.where(score[b] > 0.5)
        box_select = boxes[b][idx[:2]]
        box_xywh = box_select[:, :4]
        box_xyxy = wh2xy_np(box_xywh)
        box_truncated = []
        for box_k in box_xyxy:
            box_k[0] = box_k[0] if box_k[0] >= 0 else 0
            box_k[1] = box_k[1] if box_k[1] >= 0 else 0
            box_k[2] = box_k[2] if box_k[2] <= self.hw[1] else self.hw[1]
            box_k[3] = box_k[3] if box_k[3] <= self.hw[0] else self.hw[0]
            box_truncated.append(box_k)
        box_xyxy = np.stack(box_truncated)
        box_socre = score[b][idx]
        clsid = idx[2]
        picked_boxes = nms_np(
            np.concatenate([box_xyxy, box_socre.reshape([-1, 1]), clsid.reshape([-1, 1])], -1),
            len(self.classes))
        for bbox in picked_boxes:
            bbox[0] *= w_r
            bbox[2] *= w_r
            bbox[1] *= h_r
            bbox[3] *= h_r
            img = cv2.rectangle(img, tuple(np.int32([bbox[0], bbox[1]])),
                                tuple(np.int32([bbox[2], bbox[3]])), (0, 255, 0), 2)
            img = cv2.putText(img, "{} {:.2f}".format(self.classes[int(bbox[5])], bbox[4]),
                              tuple(np.int32([bbox[0], bbox[1]+15])),
                              cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0))
        return img

    def detect_image(self, img_path):
        img = cv2.imread(img_path)
        img = self.forward(img)
        cv2.imshow('img', img)
        cv2.waitKey(0)

    def detect_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("Couldn't open webcam or video")
        video_FourCC = int(cap.get(cv2.CAP_PROP_FOURCC))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter('output.mp4', video_FourCC, video_fps, (width, height))

        total_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        time1 = time.time()

        while True:
            ret, frame = cap.read()
            if ret:
                img = cv2.resize(frame, tuple(self.hw)[::-1])
                out = self.forward(img)
                time2 = time.time()
                d_time = time2 - time1
                time1 = time2
                total_time += d_time
                curr_fps += 1
                if total_time >= 1:
                    fps = "FPS: {}".format(curr_fps)
                    total_time -= 1
                    curr_fps = 0

                out = cv2.putText(out, fps, tuple(np.int32([20, 20])),
                                  cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
                out = cv2.resize(out, (width, height))
                cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
                cv2.imshow('result', out)
                cv2.waitKey(1)
                writer.write(out)
            else:
                break


if __name__ == '__main__':
    config = get_config()
    yolo = YOLO()
    if config.video:
        yolo.detect_video(config.video)
    elif config.image:
        yolo.detect_image(config.image)
    else:
        while True:
            img_path = input('input image path:')
            try:
                yolo.detect_image(img_path)
            except:
                print('check your iamge path ')
                continue
