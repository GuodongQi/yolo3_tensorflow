import tensorflow as tf
import numpy as np
import cv2
import time

from tensorflow.python import debug as tf_debug

from net.yolo3_tiny import model, loss
from util.image_util import read_image


class YOLO():
    def __init__(self):
        self.anchor_path = "C:\\Users\\guodong\\PycharmProjects\\egame_qq_wzry\\yolo3\\model_data\\yolo_anchors.txt"
        self.train_path = 'C:\\Users\\guodong\\PycharmProjects\\egame_qq_wzry\\yolo3\\model_data\\train.txt'
        self.classes_path = 'C:\\Users\\guodong\\PycharmProjects\\egame_qq_wzry\\yolo3\\model_data\\voc_classes.txt'
        # self.pretrain_path = 'C:\\Users\\guodong\\PycharmProjects\\egame_qq_wzry\\yolo3\\logs\\'
        self.pretrain_path = ''

        self.batch_size = 2

        self.learn_rate = 1e-3
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.iou_threshold = 0.05

        self.gt_max_size = 10

        self.classes = self._get_classes()
        self.anchors = self._get_anchors()
        self.hw = [320, 640]

        self.input = tf.placeholder(tf.float32, [self.batch_size] + self.hw + [3])
        self.label = tf.placeholder(tf.float32, [self.batch_size, self.gt_max_size, 4 + len(self.classes)])

    def _get_anchors(self):
        """loads the anchors from a file"""
        with open(self.anchor_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def _get_classes(self):
        """loads the classes"""
        with open(self.classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def create_model(self):
        pred = model(self.input, len(self.classes), self.anchors, True)
        losses = loss(pred, self.label, self.lambda_coord, self.lambda_noobj, self.iou_threshold)
        opt = tf.train.AdamOptimizer(self.learn_rate)
        op = opt.minimize(losses)
        return [pred, losses, op]

    def generate_data(self):
        with open(self.train_path) as f:
            gts = f.readlines()
        np.random.shuffle(gts)
        gts = gts[0:self.batch_size]
        img_files = []
        labels = []
        for gt in gts:
            f_path, *_label = gt.split(' ')
            if not len(_label):
                f_path = f_path.split('\n')[0]
            img, height, width = read_image(f_path)

            h_scale = 320 / height
            w_scale = 640 / width
            self.anchors *= np.array([w_scale, h_scale])

            img_files.append(img)
            gt_template = np.zeros([self.gt_max_size, 4 + len(self.classes)])
            # gt_template[:, :4] += 0.001
            for i, per_label in enumerate(_label):
                if i == self.gt_max_size:
                    break
                xmin, ymin, xmax, ymax, cls = list(map(int, per_label.split(',')))
                gt_template[i, :4] = [xmin * w_scale, ymin * h_scale, xmax * w_scale, ymax * h_scale]
                gt_template[i, 4 + cls] = 1
            labels.append(gt_template)
        return np.array(img_files, np.float32), np.array(labels, np.float32)

    def train(self):
        # pred, losses, op = self.create_model()
        pred = model(self.input, len(self.classes), self.anchors, True, 0.3)
        losses = loss(pred, self.label, self.hw, self.lambda_coord, self.lambda_noobj, self.iou_threshold)
        opt = tf.train.AdamOptimizer(self.learn_rate)
        op = opt.minimize(losses)

        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        sess = tf.Session(config=config)
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        saver = tf.train.Saver()
        if not len(self.pretrain_path):
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            saver.restore(sess, tf.train.latest_checkpoint(self.pretrain_path))

        for step in range(1000):
            t1 = time.time()
            img, label = self.generate_data()
            t2 = time.time()
            # print(t2 - t1)
            pred_, losses_, _ = sess.run([pred, losses, op], {
                self.input: img,
                self.label: label
            })

            print('step:{} loss:{}'.format(step, losses_))
        saver.save(sess, 'C:\\Users\\guodong\\PycharmProjects\\egame_qq_wzry\\yolo3\\logs\\loss{:.4f}'.format(losses_))


if __name__ == '__main__':
    YOLO().train()
