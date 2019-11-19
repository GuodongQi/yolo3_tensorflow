import os
import random
import xml.etree.ElementTree as ET
from multiprocessing import Pool

import cv2
import numpy as np
import tensorflow as tf

from util.tfrecord_utils import convert_to_example, ImageCoder

wd = os.path.dirname(os.getcwd())
class_path = os.path.join(wd, 'model_data', 'voc_classes.txt')  # change to the classes path you want to detect

width_height = (608, 608)  # the input image size
total_task = 4  # multi-process pool number
num_shards = 4000  # Number of shards in training TFRecord files
is_train = True  # whether train dataset or valid dataset

if is_train:
    image_dir = 'C:\\Users\\guodong\\dataset\\image'  # your train image dir
    annotation_dir = 'C:\\Users\\guodong\\dataset\\annotation'  # your train image annotation  dir
    save_dir = 'C:\\Users\\guodong\\dataset\\tfrecord\\train'
    tfrecord_files = 'train_w%d_h%d' % (width_height[0], width_height[1]) + '_%04d.tfrecord'
else:
    image_dir = ''  # your val image dir
    annotation_dir = ''  # your val image annotation  dir
    save_dir = 'C:\\Users\\guodong\\dataset\\tfrecord\\valid'
    tfrecord_files = 'valid_w%d_h%d' % (width_height[0], width_height[1]) + '_%04d.tfrecord'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with open(class_path) as f:
    class_names = f.readlines()
classes = [c.strip() for c in class_names]

annotation_files = os.listdir(annotation_dir)
random.shuffle(annotation_files)


def save_to_tfrecord(task_id):
    coder = ImageCoder()
    fidx = task_id
    i = task_id
    while i < len(annotation_files):
        tf_filename = os.path.join(save_dir, tfrecord_files % fidx)
        print('task: %d, starting tfrecord file %s' % (task_id, tf_filename))
        with tf.python_io.TFRecordWriter(tf_filename) as writer:
            j = 0
            while i < len(annotation_files) and j < num_shards:
                if i % (40 * total_task) == task_id:
                    # if True:
                    print('task: %d, reading annotation file %d/%d' % (task_id, i, len(annotation_files)))
                success = add_to_tfrecord(annotation_files[i], coder, writer)
                i += total_task
                if success:
                    j += 1
        fidx += total_task
    print('task: %d, done' % task_id)


def add_to_tfrecord(xml_file, coder, writer):
    # annotation_file = annotation_files[i]
    xml_file = os.path.join(annotation_dir, xml_file)
    try:
        in_file = open(xml_file, 'r')
    except:
        print("open failed {0}".format(xml_file))
    else:
        # print("open success {0}".format(image_id))
        tree = ET.parse(in_file)
        root = tree.getroot()

        boxes = []

        image_path = '%s/%s.jpg' % (image_dir, os.path.basename(xml_file).split('.')[0])
        image_data = cv2.imread(image_path)[:, :, ::-1]
        _height, _width = image_data.shape[:2]
        image_scaled = cv2.resize(image_data, width_height)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = [int(int(xmlbox.find('xmin').text) / _width * width_height[0]),
                 int(int(xmlbox.find('ymin').text) / _height * width_height[1]),
                 int(int(xmlbox.find('xmax').text) / _width * width_height[0]),
                 int(int(xmlbox.find('ymax').text) / _height * width_height[1]),
                 cls_id]
            boxes.append(b)
        if not len(boxes):
            return False

        image_decode = coder.encode_jpeg(image_scaled)
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
