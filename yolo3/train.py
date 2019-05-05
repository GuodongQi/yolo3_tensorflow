import tensorflow as tf
import numpy as np
import cv2
import time

from tensorflow.python import debug as tf_debug

from net.yolo3_tiny import model, loss
from util.box_util import xy2wh_np
from util.image_util import read_image


class YOLO():
    def __init__(self):
        self.anchor_path = "C:\\Users\\qiguodong\\PycharmProjects\\egame_qq_wzry\\yolo3\\model_data\\yolo_anchors.txt"
        self.train_path = 'C:\\Users\\qiguodong\\PycharmProjects\\egame_qq_wzry\\yolo3\\model_data\\train.txt'
        self.classes_path = 'C:\\Users\\qiguodong\\PycharmProjects\\egame_qq_wzry\\yolo3\\model_data\\voc_classes.txt'
        self.pretrain_path = 'C:\\Users\\qiguodong\\PycharmProjects\\egame_qq_wzry\\yolo3\\logs\\'
        # self.pretrain_path = ''

        self.batch_size = 8

        self.learn_rate = 1e-3
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.iou_threshold = 0.1

        self.gt_max_size = 10

        self.classes = self._get_classes()
        self.anchors = self._get_anchors()
        self.hw = [320, 640]

        self.input = tf.placeholder(tf.float32, [self.batch_size] + self.hw + [3])

        self.label = None

        with open(self.train_path) as f:
            self.gts = f.readlines()

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

    def generate_data(self, gts, grid_shape):
        img_files = []
        labels = []
        for gt in gts:
            f_path, *_label = gt.split(' ')
            if not len(_label):
                f_path = f_path.split('\n')[0]
            img, height, width = read_image(f_path, self.hw)

            h_scale = self.hw[0] / height
            w_scale = self.hw[1] / width
            self.anchors *= np.array([w_scale, h_scale])

            img_files.append(img)
            gds = []
            for g_shape in grid_shape:
                gd = np.zeros(g_shape[1:3] + [3, 4 + len(self.classes)])
                h_r = self.hw[0] / gd.shape[0]
                w_r = self.hw[1] / gd.shape[1]
                for per_label in _label:
                    xmin, ymin, xmax, ymax, cls = list(map(float, per_label.split(',')))
                    xmin, ymin, xmax, ymax = xmin * w_scale, ymin * h_scale, xmax * w_scale, ymax * h_scale
                    x0, y0, w, h = xy2wh_np([xmin, ymin, xmax, ymax])
                    i = int(np.floor(x0 / w_r))
                    j = int(np.floor(y0 / h_r))
                    k = 0
                    if gd[j, i, k, 4 + int(cls)] == 1:
                        k += 1
                        if k == 3:
                            continue
                    gd[j, i, k, 0] = x0 / w_r - i
                    gd[j, i, k, 1] = y0 / h_r - j
                    gd[j, i, k, 2] = w
                    gd[j, i, k, 3] = h
                    gd[j, i, k, 4 + int(cls)] = 1
                gds.append(gd.reshape([-1, 3, 4 + len(self.classes)]))
            labels.append(np.concatenate(gds, 0))
        img_files, labels = np.array(img_files, np.float32), np.array(labels, np.float32)
        return img_files, labels

    def train(self):
        # pred, losses, op = self.create_model()
        pred = model(self.input, len(self.classes), self.anchors, True, 0.3)
        grid_shape = [g.get_shape().as_list() for g in pred[1]]

        s = sum([g[2] * g[1] for g in grid_shape])
        self.label = tf.placeholder(tf.float32, [self.batch_size, s, 3, 4 + len(self.classes)])

        losses = loss(pred, self.label, self.hw, self.lambda_coord, self.lambda_noobj, self.iou_threshold)
        opt = tf.train.AdamOptimizer(self.learn_rate)
        op = opt.minimize(losses)
        # img, label = self.generate_data(self.gts[0:0 + self.batch_size], grid_shape)

        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        sess = tf.Session(config=config)
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        saver = tf.train.Saver()
        if not len(self.pretrain_path):
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            saver.restore(sess, tf.train.latest_checkpoint(self.pretrain_path))

        i = 0
        for step in range(1000):
            if (i + 1) * self.batch_size >= len(self.gts):
                i = 0
                np.random.shuffle(self.gts)
            t1 = time.time()
            img, label = self.generate_data(self.gts[i:i + self.batch_size], grid_shape)
            t2 = time.time()
            # print(t2 - t1)
            pred_, losses_, _ = sess.run([pred, losses, op], {
                self.input: img,
                self.label: label
            })
            i += 1
            print('step:{} loss:{}'.format(step, losses_))
            if (step + 1) % 100 == 0:
                saver.save(sess,
                           'C:\\Users\\qiguodong\\PycharmProjects\\egame_qq_wzry\\yolo3\\logs\\step{}_loss{:.4f}'.format(
                               step + 1, losses_))


if __name__ == '__main__':
    YOLO().train()
