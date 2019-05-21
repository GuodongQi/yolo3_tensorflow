import time
from os import getcwd
from os.path import join, split

import cv2
import numpy as np
import tensorflow as tf

from net.yolo3_net import model, loss
from util.box_utils import xy2wh_np, box_anchor_iou, wh2xy_np, nms_np
from util.image_utils import read_image_and_lable
from util.train_config import get_config
from util.utils import sec2time, np_sigmoid


class YOLO():
    def __init__(self):
        self.batch_size = config.batch_size
        self.epoch = config.epoch
        self.learn_rate = config.learn_rate

        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.lambda_cls = 1
        self.iou_threshold = 0.6

        self.classes = self.__get_classes()
        self.anchors = self.__get_anchors()
        self.hw = [320, 640]
        if config.tiny:
            assert 6 == len(
                self.anchors), 'model type does not match with anchors, check anchors or type param'
            self.log_path = join(getcwd(), 'logs', config.net_type + '_tiny')
        else:
            assert 9 == len(
                self.anchors), 'model type does not match with anchors, check anchors or type param'
            self.log_path = join(getcwd(), 'logs', config.net_type + '_full')
        self.pretrain_path = config.pretrain_path

        self.input = tf.placeholder(tf.float32, [self.batch_size] + self.hw + [3])

        self.label = None

        with open(config.train_path) as f:
            self.gts = f.readlines()
        val_rate = 0.01
        spl = int(val_rate * len(self.gts))

        np.random.seed(1000)
        np.random.shuffle(self.gts)
        np.random.seed(None)

        self.train_data = self.gts[spl:]
        self.val_data = self.gts[:spl]

    def __get_anchors(self):
        """loads the anchors from a file"""
        with open(config.anchor_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def __get_classes(self):
        """loads the classes"""
        with open(config.classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def generate_data(self, grid_shape, is_val=False):
        idx = 0
        if is_val:
            gts = self.val_data
        else:
            gts = self.train_data
        while True:
            img_files = []
            labels = []
            b = 0
            while idx < len(gts):  # a batch
                try:
                    res = read_image_and_lable(gts[idx + b], self.hw, self.anchors)
                except IndexError:
                    b = 0
                else:
                    if not res:
                        b += 1
                        continue

                    img, _label, _ = res

                    img_files.append(img)
                    _label_ = np.concatenate([xy2wh_np(_label[:, :4]), _label[:, 4:]], -1)  # change to xywh

                    gds = []
                    for g_id, g_shape in enumerate(grid_shape):
                        anchors = self.anchors[[g_id, g_id + 1, g_id + 2]]
                        gd = np.zeros(g_shape[1:3] + [3, 5 + len(self.classes)])
                        h_r = self.hw[0] / gd.shape[0]
                        w_r = self.hw[1] / gd.shape[1]
                        for per_label in _label_:
                            x0, y0, w, h = per_label[:4]
                            if w == 0 or h == 0:
                                continue
                            i = int(np.floor(x0 / w_r))
                            j = int(np.floor(y0 / h_r))
                            box_iou = box_anchor_iou(anchors, per_label[2:4])
                            k = np.argmax(box_iou)
                            # gd[j, i, k, 0] = x0 / w_r - i
                            # gd[j, i, k, 1] = y0 / h_r - j
                            gd[j, i, k, 0] = x0
                            gd[j, i, k, 1] = y0
                            # gd[j, i, k, 2] = np.log(w / anchors[k, 0] + 1e-15)
                            # gd[j, i, k, 3] = np.log(h / anchors[k, 1] + 1e-15)
                            gd[j, i, k, 2] = w
                            gd[j, i, k, 3] = h
                            gd[j, i, k, 4] = 1
                            gd[j, i, k, 5 + int(per_label[4])] = 1

                        gds.append(gd.reshape([-1, 3, 5 + len(self.classes)]))
                    labels.append(np.concatenate(gds, 0))
                    b += 1
                    if len(labels) == self.batch_size:
                        idx += self.batch_size
                        break
            if idx >= len(gts):
                np.random.shuffle(gts)
                idx = 0
            img_files, labels = np.array(img_files, np.float32), np.array(labels, np.float32)
            if is_val:
                yield img_files, labels
            else:
                yield img_files, labels, idx

    def train(self):
        # pred, losses, op = self.create_model()
        pred = model(self.input, len(self.classes), self.anchors, config.net_type, True, 0.3)
        grid_shape = [g.get_shape().as_list() for g in pred[1]]

        s = sum([g[2] * g[1] for g in grid_shape])
        self.label = tf.placeholder(tf.float32, [self.batch_size, s, 3, 5 + len(self.classes)])

        losses = loss(pred, self.label, self.anchors, self.hw, self.lambda_coord, self.lambda_noobj, self.lambda_cls,
                      self.iou_threshold, config.debug)
        opt = tf.train.AdamOptimizer(self.learn_rate)
        op = opt.minimize(losses)

        # summary
        writer = tf.summary.FileWriter(self.log_path)
        img_tensor = tf.placeholder(tf.float32, [self.batch_size] + self.hw + [3])
        train_loss_tensor = tf.placeholder(tf.float32)
        val_loss_tensor = tf.placeholder(tf.float32)
        tf.summary.scalar('train_loss', train_loss_tensor)
        tf.summary.scalar('val_loss', val_loss_tensor)
        tf.summary.image('img_vis', img_tensor, self.batch_size)
        summary = tf.summary.merge_all()

        conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        sess = tf.Session(config=conf)
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # sess = tf_debug.TensorBoardDebugWrapperSession(sess, "PC-DAIXILI:6001")

        saver = tf.train.Saver(max_to_keep=1)

        # init
        init = tf.global_variables_initializer()
        sess.run(init)
        if len(self.pretrain_path):
            flag = 0
            try:
                print('try to restore the whole graph')
                saver.restore(sess, self.pretrain_path)
            except:
                print('failed to restore the whole graph')
                flag = 1
            if flag:
                try:
                    print('try to restore the graph body')
                    vs = tf.trainable_variables()
                    restore_weights = [v for v in vs if 'yolo_head' not in v.name]
                    sv = tf.train.Saver(var_list=restore_weights)
                    sv.restore(sess, self.pretrain_path)
                except:
                    raise Exception('restore body faild, please check the pretained weight')

        total_step = int(np.ceil(len(self.train_data) / self.batch_size)) * self.epoch

        print('train on {} samples, val on {} samples, batch size {}, total {} epoch'.format(len(self.train_data),
                                                                                             len(self.val_data),
                                                                                             self.batch_size,
                                                                                             self.epoch))
        step = 0
        epoch = 0
        t0 = time.time()
        for data in self.generate_data(grid_shape):
            step += 1

            img, label, idx = data
            pred_, losses_, _ = sess.run([pred, losses, op], {
                self.input: img,
                self.label: label
            })
            t1 = time.time()
            print('step:{:<d}/{} epoch:{} loss:{:< .3f} ETA:{}'.format(
                step, total_step, epoch, losses_,
                sec2time((t1 - t0) * (total_step - step))))

            t0 = time.time()
            if idx == 0:
                # cal vaild_loss
                val_loss_ = 0
                val_step = 0
                for val_data in self.generate_data(grid_shape, is_val=True):
                    img, label = val_data
                    _, losses__ = sess.run([pred, losses], {
                        self.input: img,
                        self.label: label
                    })
                    val_loss_ += losses__
                    val_step += self.batch_size
                    if val_step >= len(self.val_data):
                        break
                val_loss_ /= (val_step / self.batch_size)

                # for visual
                boxes, grid = pred_
                score = np_sigmoid(boxes[..., 4:5]) * np_sigmoid(boxes[..., 5:])
                vis_img = []

                for b in range(self.batch_size):
                    idx = np.where(score[b] > 0.5)
                    box_select = boxes[b][idx[:2]]
                    box_xywh = box_select[:, :4]
                    box_xyxy = wh2xy_np(box_xywh)
                    box_socre = score[b][idx]
                    clsid = idx[2]
                    picked_boxes = nms_np(
                        np.concatenate([box_xyxy, box_socre.reshape([-1, 1]), clsid.reshape([-1, 1])], -1),
                        len(self.classes))
                    per_img = (img[b] + 1) * 128
                    for bbox in picked_boxes:
                        per_img = cv2.rectangle(per_img, tuple(np.int32([bbox[0], bbox[1]])),
                                                tuple(np.int32([bbox[2], bbox[3]])), (0, 255, 0), 2)
                        per_img = cv2.putText(per_img, "{} {:.2f}".format(self.classes[int(bbox[5])], bbox[4]),
                                              tuple(np.int32([bbox[0], bbox[1]])),
                                              cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0))

                    vis_img.append(per_img)
                ss = sess.run(summary, feed_dict={
                    img_tensor: np.array(vis_img),
                    train_loss_tensor: losses_,
                    val_loss_tensor: val_loss_
                })
                writer.add_summary(ss, step)
                saver.save(sess, join(self.log_path, split(self.log_path)[-1] + '_model')
                           )
                print('epoch:{} train_loss:{:< .3f} val_loss:{:< .3f}'.format(
                    epoch, losses_, val_loss_))
                epoch += 1
            if epoch >= self.epoch:
                break


if __name__ == '__main__':
    config = get_config()
    YOLO().train()
