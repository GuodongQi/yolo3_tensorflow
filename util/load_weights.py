# coding=utf-8
import numpy as np
import tensorflow as tf


def load_weight(var_list, file_path):
    with open(file_path, "rb") as fp:
        _ = np.fromfile(fp, dtype=np.int32, count=5)
        weights = np.fromfile(fp, dtype=np.float32)
    ptr = 0
    i = 0
    assign_ops = []
    while i < len(var_list) - 1:
        var1 = var_list[i]
        var2 = var_list[i + 1]
        # do something only if we process conv layer
        if 'cnn' in var1.name:
            # check type of next layer
            if 'batch' in var2.name:
                # load batch norm params
                gamma, beta, mean, var = var_list[i + 1:i + 5]
                batch_norm_vars = [beta, gamma, mean, var]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(tf.assign(var, var_weights, validate_shape=True))

                # we move the pointer by 4, because we loaded 4 variables
                i += 4
            elif 'cnn' in var2.name:
                # load biases
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
                assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))
                if 'yolo_head' in bias.name:  # if num_classes is not 80
                    ptr += 255
                else:
                    ptr += bias_params

                # we loaded 1 variable
                i += 1
        # we can load weights of conv layer

        shape = var1.shape.as_list()
        num_params = np.prod(shape)
        var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
        # remember to transpose to column-major
        # DarkNet conv_weights are serialized Caffe-style:
        # (out_dim, in_dim, height, width)
        # We would like to set these to Tensorflow order:
        # (height, width, in_dim, out_dim)
        var_weights = np.transpose(var_weights, (2, 3, 1, 0))
        assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))

        if 'yolo_head' in var1.name:  # if num_classes is not 80
            shape_ = shape[:3]
            shape_.append(255)
            ptr += np.prod(shape_)

        else:
            ptr += num_params

        i += 1
    assert ptr == len(weights), "load failed, please verify your weight file"
    return assign_ops
