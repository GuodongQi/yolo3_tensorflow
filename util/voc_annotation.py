import os
import random
import xml.etree.ElementTree as ET

class_path = 'model_data/voc_classes.txt'  # change to the classes path you want to detect
is_train = True  # whether train dataset or valid dataset

if is_train:
    image_dir = ''  # your train image dir
    annotation_dir = ''  # your train image annotation  dir
    gen_files = 'train.txt'
else:
    image_dir = ''  # your val image dir
    annotation_dir = ''  # your val image annotation  dir
    gen_files = 'valid.txt'

with open(class_path) as f:
    class_names = f.readlines()
classes = [c.strip() for c in class_names]

list_file_train = open(os.path.join('model_data', gen_files), 'w')

annotation_files = os.listdir(annotation_dir)
random.shuffle(annotation_files)

for i in range(0, len(annotation_files), 1):
    annotation_file = annotation_files[i]

    list_file_train.write('%s/%s.jpg' % (image_dir, annotation_file.split('.')[0]))

    xml_file = os.path.join(annotation_dir, annotation_file)
    try:
        in_file = open(xml_file, 'r')
    except:
        print("open failed {0}".format(xml_file))
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
# clean dataset
with open(os.path.join('model_data', gen_files), 'r') as f1:
    old_line = f1.readlines()
with open(os.path.join('model_data', gen_files), 'w') as f2:
    for line in old_line:
        line_ = line.split(' ')
        if len(line_) > 1:
            f2.write(line)
