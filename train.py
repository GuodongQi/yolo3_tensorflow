import time
from collections import defaultdict
from copy import deepcopy
from os import getcwd
from os.path import join, split

import numpy as np
import tensorflow as tf

from config.train_config import get_config
from net.yolo3_net import loss, model
from util.box_utils import box_anchor_iou, pick_box, xy2wh_np
from util.image_utils import get_color_table, plot_img, read_image_and_lable
from util.utils import sec2time, cal_fp_fn_tp_tn, cal_mAP


class YOLO():
    def __init__(self, config):
        self.config = config

        self.batch_size = self.config.batch_size
        self.epoch = self.config.epoch
        self.learn_rate = self.config.learn_rate

        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.lambda_cls = 1
        self.iou_threshold = 0.5  # used to decide whether box is BG or FG

        self.ious_thres = [0.5, 0.75]  # used to calculate mAP

        self.classes = self.__get_classes()
        self.anchors = self.__get_anchors()
        self.hw = [416, 416]
        if self.config.tiny:
            assert 6 == len(
                self.anchors), 'model type does not match with anchors, check anchors or type param'
            self.log_path = join(getcwd(), 'logs', self.config.net_type + '_tiny')
        else:
            assert 9 == len(
                self.anchors), 'model type does not match with anchors, check anchors or type param'
            self.log_path = join(getcwd(), 'logs', self.config.net_type + '_full')
        self.pretrain_path = self.config.pretrain_path

        self.input = tf.placeholder(tf.float32, [self.batch_size] + self.hw + [3])
        self.is_training = tf.placeholder(tf.bool, shape=[])
        self.label = None

        with open(self.config.train_path) as f:
            self.train_data = f.readlines()
        with open(self.config.valid_path) as f:
            self.val_data = f.readlines()

        self.color_table = get_color_table(len(self.classes))

    def __get_anchors(self):
        """loads the anchors from a file"""
        with open(self.config.anchor_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def __get_classes(self):
        """loads the classes"""
        with open(self.config.classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def generate_data(self, grid_shape, is_val=False):

        gds_init = [np.zeros(g_shape[1:3] + [3, 9 + len(self.classes)]) for g_shape in grid_shape]

        idx = 0

        GTS = defaultdict(lambda: defaultdict(list))

        if is_val:
            gts = self.val_data
        else:
            gts = self.train_data
        while True:
            img_files = []
            labels = []
            b = 0
            GTS.clear()

            while idx < len(gts) - self.batch_size:  # a batch
                try:
                    res = read_image_and_lable(gts[idx + b], self.hw, is_training=not is_val)
                    # print(idx + b)
                except IndexError:
                    raise Exception('it should not happen')
                else:
                    if not res:
                        raise Exception('check your dataset, it has none label')

                    img, _label = res

                    img_files.append(img)

                    for per_xyxy in _label:
                        GTS[b][self.classes[int(per_xyxy[4])]].append(per_xyxy[:4].tolist())

                    _label_ = np.concatenate([xy2wh_np(_label[:, :4]), _label[:, 4:]], -1)  # change to xywh

                    gds = deepcopy(gds_init)
                    for per_label in _label_:
                        x0, y0, w, h = per_label[:4]
                        if w == 0 or h == 0:
                            continue
                        box_iou = box_anchor_iou(self.anchors, per_label[2:4])
                        k = np.argmax(box_iou)
                        div, mod = divmod(int(k), 3)
                        div = len(grid_shape) - 1 - div
                        h_r = self.hw[0] / gds[div].shape[0]
                        w_r = self.hw[1] / gds[div].shape[1]
                        i = int(np.floor(x0 / w_r))
                        j = int(np.floor(y0 / h_r))

                        gds[div][j, i, mod, 0] = x0 / w_r - i
                        gds[div][j, i, mod, 1] = y0 / h_r - j
                        gds[div][j, i, mod, 2] = np.log(w / self.anchors[k, 0] + 1e-5)
                        gds[div][j, i, mod, 3] = np.log(h / self.anchors[k, 1] + 1e-5)

                        gds[div][j, i, mod, 4] = x0
                        gds[div][j, i, mod, 5] = y0
                        gds[div][j, i, mod, 6] = w
                        gds[div][j, i, mod, 7] = h

                        gds[div][j, i, mod, 8] = 1
                        gds[div][j, i, mod, 9 + int(per_label[4])] = 1

                    gds = [gd.reshape([-1, 3, 9 + len(self.classes)]) for gd in gds]
                    labels.append(np.concatenate(gds, 0))
                    b += 1
                    if len(labels) == self.batch_size:
                        idx += self.batch_size
                        break
            if idx >= len(gts) - self.batch_size:
                np.random.shuffle(gts)
                idx = 0
            img_files, labels = np.array(img_files, np.float32), np.array(labels, np.float32)
            if is_val:
                yield img_files, labels, GTS
            else:
                yield img_files, labels, idx

    def train(self):
        # pred, losses, op = self.create_model()
        pred = model(self.input, len(self.classes), self.anchors, self.config.net_type, self.is_training, True)
        grid_shape = [g.get_shape().as_list() for g in pred[2]]

        s = sum([g[2] * g[1] for g in grid_shape])
        self.label = tf.placeholder(tf.float32, [self.batch_size, s, 3, 9 + len(self.classes)])
        # for data in self.generate_data(grid_shape):
        #     print()

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        var_list = tf.global_variables()

        losses = loss(pred, self.label, self.hw, self.lambda_coord, self.lambda_noobj, self.lambda_cls,
                      self.iou_threshold, self.config.debug)
        opt = tf.train.AdamOptimizer(self.learn_rate)

        with tf.control_dependencies(update_ops):
            op = opt.minimize(losses)

        # summary
        writer = tf.summary.FileWriter(self.log_path, max_queue=-1)
        img_tensor = tf.placeholder(tf.float32, [2 * self.batch_size] + self.hw + [3])

        with tf.name_scope('loss'):
            train_loss_tensor = tf.placeholder(tf.float32)
            val_loss_tensor = tf.placeholder(tf.float32)
            tf.summary.scalar('train_loss', train_loss_tensor)
            tf.summary.scalar('val_loss', val_loss_tensor)

        with tf.name_scope('mAP'):
            for iou in self.ious_thres:
                with tf.name_scope('iou{}'.format(iou)):
                    exec('map_with_iou{} = tf.placeholder(tf.float32)'.format(int(iou * 100)))
                    exec('tf.summary.scalar("mAP", map_with_iou{})'.format(int(iou * 100)))

        with tf.name_scope('per_class_AP'):
            for iou in self.ious_thres:
                with tf.name_scope('iou{}'.format(iou)):
                    for per_cls in self.classes:
                        per_cls = per_cls.replace(' ', '_')
                        exec('ap_{}_with_iou{} = tf.placeholder(tf.float32)'.format(per_cls, int(iou * 100)))
                        exec('tf.summary.scalar("{}", ap_{}_with_iou{})'.format(per_cls, per_cls, int(iou * 100)))

        tf.summary.image('img', img_tensor, 2 * self.batch_size)
        summary = tf.summary.merge_all()

        conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        sess = tf.Session(config=conf)
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # sess = tf_debug.TensorBoardDebugWrapperSession(sess, "PC-DAIXILI:6001")

        saver = tf.train.Saver(var_list=var_list, max_to_keep=5)
        # saver = tf.train.Saver()

        # init
        init = tf.global_variables_initializer()
        sess.run(init)

        if len(self.pretrain_path):
            flag = 0
            try:
                print('try to restore the whole graph')
                saver.restore(sess, self.pretrain_path)
                print('successfully restore the whole graph')
            except:
                print('failed to restore the whole graph')
                flag = 1
            if flag:
                try:
                    print('try to restore the graph body')
                    restore_weights = [v for v in var_list if 'yolo_head' not in v.name]
                    sv = tf.train.Saver(var_list=restore_weights)
                    sv.restore(sess, self.pretrain_path)
                    print('successfully restore the graph body')
                except Exception:
                    raise Exception('restore body failed, please check the pretained weight')

        total_step = int(np.ceil(len(self.train_data) / self.batch_size)) * self.epoch

        print('train on {} samples, val on {} samples, batch size {}, total {} epoch'.format(len(self.train_data),
                                                                                             len(self.val_data),
                                                                                             self.batch_size,
                                                                                             self.epoch))
        step = 0
        epoch = 0
        t0 = time.time()

        DETECTION = defaultdict(lambda: defaultdict(list))
        FP_TP = defaultdict(lambda: defaultdict(list))
        GT_NUMS = defaultdict(int)

        for data in self.generate_data(grid_shape):
            step += 1

            img, label, idx = data
            pred_, losses_, _ = sess.run([pred, losses, op], {
                self.input: img,
                self.label: label,
                self.is_training: True
            })
            t1 = time.time()
            print('step:{:<d}/{} epoch:{} loss:{:< .3f} ETA:{}'.format(
                step, total_step, epoch, losses_,
                sec2time((t1 - t0) * (total_step / step - 1))))

            if idx == 0:
                # for training visual
                raw_, boxes, grid = pred_
                vis_img = []
                for b in range(self.batch_size):
                    picked_boxes = pick_box(boxes[b], 0.3, 0.3, self.hw, self.classes)
                    per_img = np.array(img[b] * 255, dtype=np.uint8)
                    # draw pred
                    per_img_ = per_img.copy()
                    per_img_ = plot_img(per_img_, picked_boxes, self.color_table, self.classes)
                    vis_img.append(per_img_)

                    # draw gts
                    per_img_ = per_img.copy()
                    per_label = label[b]
                    picked_boxes = pick_box(per_label[..., 4:], 0.3, 0.3, self.hw, self.classes)
                    per_img_ = plot_img(per_img_, picked_boxes, self.color_table, self.classes,
                                        True)
                    vis_img.append(per_img_)

                # cal valid_loss
                val_loss_ = 0
                val_step = 0

                cnt = 0

                GT_NUMS.clear()
                DETECTION.clear()
                FP_TP.clear()

                for val_data in self.generate_data(grid_shape, is_val=True):

                    cnt += self.batch_size
                    print("valid data: {}/{}".format(cnt, len(self.val_data)), end='\n')
                    img, label, GTS = val_data
                    pred_, losses__ = sess.run([pred, losses], {
                        self.input: img,
                        self.label: label,
                        self.is_training: False
                    })

                    _, boxes_, _ = pred_
                    for b in range(self.batch_size):
                        DETECTION[b] = defaultdict(list)
                        picked_boxes = pick_box(boxes_[b], 0.01, 0.5, self.hw, self.classes)  # NMS
                        for picked_box in picked_boxes:
                            DETECTION[b][self.classes[int(picked_box[5])]].append(picked_box[:5].tolist())

                    # cal FP TP
                    # import pdb
                    # pdb.set_trace()
                    cal_fp_fn_tp_tn(DETECTION, GTS, FP_TP, GT_NUMS, self.classes, self.ious_thres)

                    val_loss_ += losses__
                    val_step += self.batch_size

                    DETECTION.clear()

                    if val_step >= len(self.val_data):
                        break

                APs, mAPs = cal_mAP(FP_TP, GT_NUMS, self.classes, self.ious_thres)
                print(APs)
                print(mAPs)
                # import pdb
                # pdb.set_trace()
                val_loss_ /= (val_step / self.batch_size)

                feed_dict = {
                    img_tensor: np.array(vis_img),
                    train_loss_tensor: losses_,
                    val_loss_tensor: val_loss_
                }

                for iou in self.ious_thres:
                    exec('feed_dict[map_with_iou{0}] = mAPs[{1}] '.format(int(iou * 100), iou))
                    for per_cls in self.classes:
                        per_clses = per_cls.replace(' ', '_')
                        exec(
                            'feed_dict[ap_{0}_with_iou{1}] = APs[{2}]["{3}"] '.format(per_clses, int(iou * 100), iou,
                                                                                      per_cls))

                ss = sess.run(summary, feed_dict=feed_dict)
                writer.add_summary(ss, epoch)
                saver.save(sess, join(self.log_path, split(self.log_path)[-1] + '_model_epoch_{}'.format(epoch)),
                           write_meta_graph=False, write_state=False)
                print('epoch:{} train_loss:{:< .3f} val_loss:{:< .3f}'.format(
                    epoch, losses_, val_loss_))
                epoch += 1
            if epoch >= self.epoch:
                break


if __name__ == '__main__':
    configs = get_config()
    YOLO(configs).train()
