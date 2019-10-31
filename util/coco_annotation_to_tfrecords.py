from collections import defaultdict

import cv2
import json
import numpy as np
import os
import tensorflow as tf
from multiprocessing import Pool

from util.tfrecord_utils import convert_to_example, ImageCoder

wd = os.path.dirname(os.getcwd())
class_path = os.path.join(wd, 'model_data', 'coco_classes.txt')  # change to the classes path you want to detect

width_height = (608, 608)  # the input image size
total_task = 10  # multi-process pool number
num_shards = 4000  # Number of shards in training TFRecord files
is_train = True  # whether train dataset or valid dataset

if is_train:
    image_dir = '/media/data1/datasets/Generic/coco/train2017'  # your train image dir
    annotation_file = '/media/data1/datasets/Generic/coco/annotations/instances_train2017.json'  # your train image annotation  dir
    save_dir = '/media/data2/qiguodong/new/tfrecords/train'
    tfrecord_files = 'train_w%d_h%d' % (width_height[0], width_height[1]) + '_%04d.tfrecord'
else:
    image_dir = '/media/data1/datasets/Generic/coco/val2017'  # your val image dir
    annotation_file = '/media/data1/datasets/Generic/coco/annotations/instances_val2017.json'  # your val image annotation  dir
    save_dir = '/media/data2/qiguodong/new/tfrecords/valid'
    tfrecord_files = 'valid_w%d_h%d' % (width_height[0], width_height[1]) + '_%04d.tfrecord'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

name_box_id = defaultdict(list)
id_name = dict()

with open(class_path) as f:
    class_names = f.readlines()
classes = [c.strip() for c in class_names]

with open(annotation_file) as f:
    data = json.load(f)
annotations = data['annotations']

for ant in annotations:
    image_id = ant['image_id']
    image_path = os.path.join(image_dir, '%012d.jpg' % image_id)
    cat = ant['category_id']

    if 1 <= cat <= 11:
        cat -= 1
    elif 13 <= cat <= 25:
        cat -= 2
    elif 27 <= cat <= 28:
        cat -= 3
    elif 31 <= cat <= 44:
        cat -= 5
    elif 46 <= cat <= 65:
        cat -= 6
    elif cat == 67:
        cat -= 7
    elif cat == 70:
        cat -= 9
    elif 72 <= cat <= 82:
        cat -= 10
    elif 84 <= cat <= 90:
        cat -= 11
    name_box_id[image_path].append([ant['bbox'], cat])

keys = list(name_box_id.keys())


def save_to_tfrecord(task_id):
    coder = ImageCoder()
    fidx = task_id
    i = task_id
    while i < len(name_box_id):
        tf_filename = os.path.join(save_dir, tfrecord_files % fidx)
        print('task: %d, starting tfrecord file %s' % (task_id, tf_filename))
        with tf.python_io.TFRecordWriter(tf_filename) as writer:
            j = 0
            while i < len(name_box_id) and j < num_shards:
                if i % (40 * total_task) == task_id:
                    # if True:
                    print('task: %d, reading annotation file %d/%d' % (task_id, i, len(name_box_id)))
                img_path = keys[i]
                success = add_to_tfrecord(img_path, name_box_id[img_path], coder, writer)
                i += total_task
                if success:
                    j += 1
        fidx += total_task
    print('task: %d, done' % task_id)


def add_to_tfrecord(img_path, box_info, coder, writer):
    # annotation_file = annotation_files[i]

    image_path = os.path.join(image_dir, img_path)
    image_data = cv2.imread(image_path)[:, :, ::-1]
    image_scaled = cv2.resize(image_data, width_height)
    image_decode = coder.encode_jpeg(image_scaled)
    boxes = []
    for info in box_info:
        x_min = int(info[0][0])
        y_min = int(info[0][1])
        x_max = x_min + int(info[0][2])
        y_max = y_min + int(info[0][3])
        boxes.append([x_min, y_min, x_max, y_max, int(info[1])])

    label = np.hstack(boxes)
    example = convert_to_example(image_decode, image_path, width_height[1], width_height[0], label)
    writer.write(example.SerializeToString())
    return True


if __name__ == '__main__':
    p = Pool(total_task)
    for i in range(total_task):
        p.apply_async(save_to_tfrecord, [i])
    p.close()
    p.join()
