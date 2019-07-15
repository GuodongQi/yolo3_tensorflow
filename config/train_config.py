import argparse
from os import getcwd
from os.path import join


def str2bool(v):
    if v.lower() in ['yes', 'true']:
        return True
    elif v.lower() in ['no', 'false']:
        return False
    else:
        raise argparse.ArgumentTypeError()


def get_config():
    root = getcwd()
    conf = argparse.ArgumentParser()

    # yolo3 type
    conf.add_argument('-n', "--net_type", type=str, help='net type: cnn, mobilenetv1 mobilenetv2 or mobilenetv3',
                      default='cnn')
    conf.add_argument('-t', '--tiny', type=str2bool, help='whether tiny yolo or not', default=False)

    # training argument
    conf.add_argument('-b', '--batch_size', type=int, help='batch_size', default=4)
    conf.add_argument('-e', '--epoch', type=int, help='epoch', default=100)
    conf.add_argument('-lr', '--learn_rate', type=float, help='learn_rate', default=1e-4)

    # load pretrain
    conf.add_argument('-pt', '--pretrain_path', type=str, help='pretrain path', default='')

    conf.add_argument('--anchor_path', type=str, help='anchor path',
                      default=join(root, 'model_data', 'yolo_anchors.txt'))
    conf.add_argument('--train_path', type=str, help='train file path',
                      default=join(root, 'model_data', 'train.txt'))
    conf.add_argument('--valid_path', type=str, help='valid file path',
                      default=join(root, 'model_data', 'valid.txt'))
    conf.add_argument('--classes_path', type=str, help='classes path',
                      default=join(root, 'model_data', 'voc_classes.txt'))

    conf.add_argument('-d', '--debug', type=str2bool, help='whether print per item loss', default=False)
    return conf.parse_args()
