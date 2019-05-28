# coding=utf8

import random

import cv2
import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv


def rand(a=0., b=1.):
    return random.random() * (b - a) + a


def read_image_and_lable(gt_path, hw, anchor, hue=.1, sat=1.5, val=1.5):
    """read image form image_set path random distort image """
    f_path, *_label = gt_path.split(' ')
    if not len(_label):
        # f_path = f_path.split('\n')[0]
        return
    image_raw_data = cv2.imread(f_path)[..., ::-1]  # RGB h*w*c
    height, width = image_raw_data.shape[0], image_raw_data.shape[1]
    image_data = cv2.resize(image_raw_data, tuple(hw[::-1])) / 255.0

    h_scale = hw[0] / height
    w_scale = hw[1] / width
    # anchor[:, 0] *= w_scale
    # anchor[:, 1] *= h_scale

    xyxy = []

    for per_label in _label:
        xmin, ymin, xmax, ymax, cls = list(map(float, per_label.split(',')))
        xyxy.append([xmin * w_scale, ymin * h_scale, xmax * w_scale, ymax * h_scale, cls])
    xyxy = np.array(xyxy)

    # random flip image from top to down
    if rand() < .5:
        image_data = cv2.flip(image_data, 0)
        tmp = xyxy[:, 1].copy()
        xyxy[:, 1] = hw[0] - xyxy[:, 3]
        xyxy[:, 3] = hw[0] - tmp

    # random flip image from left to right
    if rand() < .5:
        image_data = cv2.flip(image_data, 1)
        tmp = xyxy[:, 0].copy()
        xyxy[:, 0] = hw[1] - xyxy[:, 2]
        xyxy[:, 2] = hw[1] - tmp

    # distort image
    if rand() < 0.5:
        x = rgb_to_hsv(image_data)
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x[..., 0] += hue
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x > 1] = 1
        x[x < 0] = 0

        image_data = hsv_to_rgb(x)  # RGB
    # random pad
    if rand() < .5:
        pad_top = random.randint(0, 25)
        pad_left = random.randint(0, 25)
        if rand() < .5:
            image_data = np.pad(image_data, ((pad_top, 0), (pad_left, 0), (0, 0)), 'edge')
        else:
            image_data = np.pad(image_data, ((pad_top, 0), (pad_left, 0), (0, 0)), 'constant')
        image_data = image_data[:hw[0], :hw[1], :]
        for i in range(xyxy.shape[0]):
            xyxy[i, 0] = pad_left + xyxy[i, 0] if pad_left + xyxy[i, 0] < hw[1] else hw[1]
            xyxy[i, 2] = pad_left + xyxy[i, 2] if pad_left + xyxy[i, 2] < hw[1] else hw[1]
            xyxy[i, 1] = pad_top + xyxy[i, 1] if pad_top + xyxy[i, 1] < hw[0] else hw[0]
            xyxy[i, 3] = pad_top + xyxy[i, 3] if pad_top + xyxy[i, 3] < hw[0] else hw[0]
    # random pad
    if rand() < .5:
        pad_bottom = random.randint(0, 25)
        pad_right = random.randint(0, 25)
        if rand() < .5:
            image_data = np.pad(image_data, ((0, pad_bottom), (0, pad_right), (0, 0)), 'edge')
        else:
            image_data = np.pad(image_data, ((0, pad_bottom), (0, pad_right), (0, 0)), 'constant')
        image_data = image_data[pad_bottom:hw[0] + pad_bottom, pad_right:hw[1] + pad_right, :]
        for i in range(xyxy.shape[0]):
            xyxy[i, 0] = xyxy[i, 0] - pad_right if xyxy[i, 0] - pad_right > 0 else 0
            xyxy[i, 2] = xyxy[i, 2] - pad_right if xyxy[i, 2] - pad_right > 0 else 0
            xyxy[i, 1] = xyxy[i, 1] - pad_bottom if xyxy[i, 1] - pad_bottom > 0 else 0
            xyxy[i, 3] = xyxy[i, 3] - pad_bottom if xyxy[i, 3] - pad_bottom > 0 else 0
    return image_data, xyxy, anchor


def get_color_table(class_num, seed=200):
    random.seed(seed)
    color_table = {}
    for i in range(class_num):
        color_table[i] = [random.randint(0, 255) for _ in range(3)]
    return color_table


def plot_rectangle(img, picked_boxes, w_r, h_r, color_table, classes, is_gt=False):
    for co, bbox in enumerate(picked_boxes):
        bbox[0] *= w_r
        bbox[2] *= w_r
        bbox[1] *= h_r
        bbox[3] *= h_r
        color = color_table[int(bbox[5])]
        tl = int(min(round(0.002 * max(img.shape[0:2])), min(bbox[3] - bbox[1], bbox[2] - bbox[0])))
        t2 = max(tl - 1, 1)  # font thickness
        if is_gt:
            label = "gts: {}".format(classes[int(bbox[5])])
        else:
            label = "{} {:.2f}".format(classes[int(bbox[5])], bbox[4])
        img = cv2.rectangle(img, tuple(np.int32([bbox[0], bbox[1]])),
                            tuple(np.int32([bbox[2], bbox[3]])), color, 2)
        img = cv2.putText(img, label, tuple(np.int32([bbox[0], bbox[1] + 15])),
                          cv2.FONT_HERSHEY_TRIPLEX, float(tl) / 3, color, thickness=t2, lineType=cv2.LINE_AA)
    return img
