import xml.etree.ElementTree as ET
import os
import random

classes = ['blood']

# wd = os.path.dirname(os.getcwd())

wd = 'C:\\Users\\qiguodong\\'

list_file_train = open('C:\\Users\\qiguodong\\PycharmProjects\\egame_qq_wzry\\yolo3\\model_data\\train.txt', 'w')

annotation_files = os.listdir(os.path.join(wd, 'dataset', 'annotation'))
random.shuffle(annotation_files)

for i in range(0, len(annotation_files), 1):
    annotation_file = annotation_files[i]

    list_file_train.write('%s\\dataset\\image\\%s.jpg' % (wd, annotation_file.split('.')[0]))

    mypath = os.path.join(wd,'dataset', 'annotation', annotation_file)
    try:
        in_file = open(mypath, 'r')
    except:
        print("open failed {0}".format(mypath))
    else:
        # print("open success {0}".format(image_id))
        tree = ET.parse(in_file)
        root = tree.getroot()

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
                 int(xmlbox.find('ymax').text))
            # list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
            list_file_train.write(" " + ",".join([str(a) for a in b]) + ',' + str(0))
    list_file_train.write('\n')

list_file_train.close()
# list_file_val.close()
