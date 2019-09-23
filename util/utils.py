import numpy as np

from util.box_utils import box_iou_np
from collections import defaultdict


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sec2time(sec, n_msec=3):
    ''' Convert seconds to 'D days, HH:MM:SS.FFF' '''
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if n_msec > 0:
        pattern = '%%02dh %%02dm %%0%d.%dfs' % (n_msec + 3, n_msec)
    else:
        pattern = r'%02dh %02dm %02s'
    if d == 0:
        return pattern % (h, m, s)
    return ('%d d, ' + pattern) % (d, h, m, s)


def cal_fp_fn_tp_tn(detection, ground_truth, FP_TP, GT_NUM, classes, iou_thres_list):
    """
    calculate FP TP FN TN accroding to detection and ground truth
    :param detection: a dict, the format:

    {
        image1: { class1: [
                           [xmin, ymim, xmax, ymax, confidence_score], # obj1
                           [xmin, ymim, xmax, ymax, confidence_score], # obj2
                           ...
                           ],
                  class2: [[xmin, ymim, xmax, ymax, confidence_score]],
                  ...
                },

        image2: { class1: [
                           [xmin, ymim, xmax, ymax, confidence_score], # obj1
                           [xmin, ymim, xmax, ymax, confidence_score] # obj2
                           ...
                           ],
                ...
                },
        ...
    }


    :param ground_truth: a dict:
    {
        image1: { class1: [
                           [xmin, ymim, xmax, ymax], # obj1
                           [xmin, ymim, xmax, ymax] # obj2
                           ...
                           ],
                  class2: [
                           [xmin, ymim, xmax, ymax], # obj1
                           [xmin, ymim, xmax, ymax] # obj2
                           ...
                           ],
                   ...
                },

        image2: { class1: [
                           [xmin, ymim, xmax, ymax], # obj1
                           [xmin, ymim, xmax, ymax] # obj2
                           ...
                           ],
                ...
                },
        ...
    }

    :param FP_TP : a dict returned
    {
        iou_thres1: {
                class1: [
                         [False, confidence_score],  # image1_obj1, False means FP, True means TP
                         [False, confidence_score],  # image1_obj2, False means FP, True means TP
                         [False, confidence_score],  # image2_obj1, False means FP, True means TP
                         [False, confidence_score],  # image2_obj2, False means FP, True means TP
                         ],
                class2: [[False, confidence_score]],
                      ...
                },

        iou_thres2: {
                class1: [
                         [False, confidence_score],  # image1_obj1, False means FP, True means TP
                         [False, confidence_score],  # image1_obj2, False means FP, True means TP
                         [False, confidence_score],  # image2_obj1, False means FP, True means TP
                         [False, confidence_score],  # image2_obj2, False means FP, True means TP
                         ],
                class2: [[False, confidence_score]],
                      ...
                },
        ...
    }


    :param GT_NUM: a dict that stores the total gt box, to calculate recall rate
    {
        class1: num1,
        class2: num2,
        ...
    }

    :param classes: list, classes name
    :param iou_thres_list: list, iou_threshold


    """
    for i in detection.keys():  # image file name
        det_objs = detection[i]  # detection dict
        gt_objs = ground_truth[i]  # gt dict
        for j in classes:  # class name
            det_boxes = np.array(det_objs[j])  # detection boxes
            gt_boxes = np.array(gt_objs[j])  # gt boxes

            if not len(gt_boxes):  # if gt boxes is none, all detection box is FP
                for iou_thres in iou_thres_list:
                    for box_index, box in enumerate(det_boxes):  # init
                        FP_TP[iou_thres][j].append([False, box[4]])
                continue

            GT_NUM[j] += len(gt_boxes)

            if not len(det_boxes):  # if gt boxes is not none, but detection box is NONE, only add the gt num
                continue

            ious = box_iou_np(det_boxes, gt_boxes)  # calculate iou
            # ious_larger = np.where(ious > iou_thres, ious, np.zeros_like(ious))
            ious_index = np.argmax(ious, 0)  # find max iou index, which will be TP, others will be FP

            for iou_thres in iou_thres_list:
                for box_index, box in enumerate(det_boxes):  # init
                    FP_TP[iou_thres][j].append([False, box[4]])

                for gt_index in range(len(gt_boxes)):
                    selected = ious_index[gt_index]
                    sel_index = len(det_boxes) - selected - 1
                    FP_TP[iou_thres][j][~sel_index][0] = ious[selected, gt_index] >= iou_thres


def cal_mAP(FP_TP, GT_NUM, classes, iou_thres_list):
    """
    calculate mAP
    :param FP_TP : a dict returned
    {
        iou_thres1: {
                class1: [
                         [False, confidence_score],  # image1_obj1, False means FP, True means TP
                         [False, confidence_score],  # image1_obj2, False means FP, True means TP
                         [False, confidence_score],  # image2_obj1, False means FP, True means TP
                         [False, confidence_score],  # image2_obj2, False means FP, True means TP
                         ],
                class2: [[False, confidence_score]],
                      ...
                },

        iou_thres2: {
                class1: [
                         [False, confidence_score],  # image1_obj1, False means FP, True means TP
                         [False, confidence_score],  # image1_obj2, False means FP, True means TP
                         [False, confidence_score],  # image2_obj1, False means FP, True means TP
                         [False, confidence_score],  # image2_obj2, False means FP, True means TP
                         ],
                class2: [[False, confidence_score]],
                      ...
                },
        ...
    }


    :param GT_NUM: a dict that stores the total gt box, to calculate recall rate
    {
        class1: num1,
        class2: num2,
        ...
    }

    :param classes: list, classes name
    :param iou_thres_list: list, iou_threshold

    """
    iou_class_AP = {}
    iou_mAP = {}
    for iou_thres in iou_thres_list:
        class_AP = {}
        for cls in classes:
            fp_tp = FP_TP[iou_thres][cls]
            fp_tp = sorted(fp_tp, key=lambda x: x[1], reverse=True)
            TP, total_det = 0, 0
            precision = [1.0]
            recall = [0.0]

            # calculate pr for each box
            for per_fp_tp in fp_tp:
                total_det += 1
                if per_fp_tp[0]:
                    TP += 1
                precision.append(TP / total_det)
                if not GT_NUM[cls]:
                    print('your valid or test data is too small that cannot cover all classes')
                    recall.append(0)
                else:
                    recall.append(TP / GT_NUM[cls])

            # calculate AP by all points interpolation
            AP = 0
            i_old = 0
            for i in range(1, len(recall)):
                if recall[i] == recall[i_old]:
                    continue
                p = max(precision[i:])
                AP += p * (recall[i] - recall[i_old])
                i_old = i
            class_AP[cls] = AP
        iou_class_AP[iou_thres] = class_AP
        iou_mAP[iou_thres] = sum(class_AP.values()) / len(classes)

    return iou_class_AP, iou_mAP


if __name__ == '__main__':
    detection = {
        "image1": {
            "class1": [[1, 2, 3, 4, 5],
                       [10, 20, 30, 40, 4],
                       [1, 2, 3, 40, 3],
                       ],
            "class2": [[1, 2, 3, 4, 5],
                       [10, 20, 30, 40, 4],
                       [1, 2, 3, 40, 3],
                       ]
        },
        "image2": {
            "class1": [[1, 2, 3, 4, 5],
                       [10, 20, 30, 40, 4],
                       [1, 2, 3, 40, 3],
                       ],
            "class2": [[1, 2, 3, 4, 5],
                       [10, 20, 30, 40, 4],
                       [1, 2, 3, 40, 3],
                       ]
        }

    }

    GT = {
        "image1": {
            "class1": [[1, 2, 3.4, 4],
                       [10, 20, 30, 40],
                       ],
            "class2": [[1, 2, 3.4, 4],
                       ],
        },
        "image2": {
            "class1": [[1, 2, 3.4, 4],
                       [10, 20, 30, 40],
                       ],
            "class2": [[1, 2, 3.4, 4],
                       ],
        },
    }
    fp = defaultdict(lambda: defaultdict(list))
    nums = defaultdict(int)
    cal_fp_fn_tp_tn(detection, GT, fp, nums, ["class1", "class2"], [0.4, 0.9, 1])
    a = cal_mAP(fp, nums, ["class1", "class2"], [0.4, 0.9, 1])
    print()
