import argparse
from os import getcwd
from os.path import join


def get_config():
    root = getcwd()
    conf = argparse.ArgumentParser()

    conf.add_argument('-i', '--image', default=None, type=str, help='image path')
    conf.add_argument('-v', '--video', default=None, type=str, help='video path')

    # load weight_path
    conf.add_argument('-w', '--weight_path', type=str, help='weight path',
                      default='logs/cnn_full/cnn_full_model_epoch_20')

    conf.add_argument('--score', default=0.3, type=float, help='score threshold')

    conf.add_argument('--classes_path', type=str, help='classes path',
                      default=join(root, 'model_data', 'coco_classes.txt'))

    return conf.parse_args()
