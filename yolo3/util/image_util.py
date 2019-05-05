# coding=utf8

import cv2
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import random


def rand(a=0., b=1.):
    return random.random() * (b - a) + a


def read_image(image_path, hw, hue=.1, sat=1.5, val=1.5):
    """read image form image_set path random distort image """

    image_raw_data = cv2.imread(image_path)[..., ::-1]  # RGB h*w*c
    height, width = image_raw_data.shape[0], image_raw_data.shape[1]
    image_data = cv2.resize(image_raw_data, tuple(hw[::-1]))
    # random flip image from top to down
    # if rand() < .5:
    #     image_raw_data = cv2.flip(image_raw_data, 0)

    # distort image
    distort = True
    if distort:
        x = rgb_to_hsv(image_data / 255)
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

        image_data = hsv_to_rgb(x) * 2 - 1  # RGB
    else:
        image_data = image_data / 128 - 1

    # image_data = image_data / 127.0 - 1
    # image_data = tf.expand_dims(image_data, 0)
    # image_data_set.append(image_data)

    # image_data_set = np.stack(image_data_set, 0)
    # return image_data_set.eval(session=sess) / 255
    return image_data, height, width

# def get_true_gts(gts,grid_shape):
