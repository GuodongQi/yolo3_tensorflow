import json
import os
from collections import defaultdict

wd = os.path.dirname(os.getcwd())
class_path = os.path.join(wd, 'model_data', 'coco_classes.txt')  # change to the classes path you want to detect
is_train = 0  # whether train dataset or valid dataset

if is_train:
    image_dir = 'F:\\coco\\train2014'  # your train image dir
    annotation_file = 'F:\\coco\\annotations\\instances_train2014.json'  # your train image annotation  dir
    prefix = 'COCO_train2014'
    gen_files = 'train.txt'
else:
    image_dir = 'F:\\coco\\val2014'  # your val image dir
    annotation_file = 'F:\\coco\\annotations\\instances_val2014.json'  # your val image annotation  dir
    prefix = 'COCO_val2014'
    gen_files = 'valid.txt'

name_box_id = defaultdict(list)
id_name = dict()
with open(class_path) as f:
    class_names = f.readlines()
classes = [c.strip() for c in class_names]

list_file = open(os.path.join(wd, 'model_data', gen_files), 'w')

with open(annotation_file) as f:
    data = json.load(f)
annotations = data['annotations']

for ant in annotations:
    image_id = ant['image_id']
    image_path = os.path.join(image_dir, prefix + '_%012d.jpg' % image_id)
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

for key, box_infos in name_box_id.items():
    list_file.write(key)
    for info in box_infos:
        x_min = int(info[0][0])
        y_min = int(info[0][1])
        x_max = x_min + int(info[0][2])
        y_max = y_min + int(info[0][3])

        box_info = " %d,%d,%d,%d,%d" % (x_min, y_min, x_max, y_max, int(info[1]))
        list_file.write(box_info)
    list_file.write('\n')
list_file.close()

# list_file_val.close()
# clean dataset
with open(os.path.join(wd, 'model_data', gen_files), 'r') as f1:
    old_line = f1.readlines()
with open(os.path.join(wd, 'model_data', gen_files), 'w') as f2:
    for line in old_line:
        line_ = line.split(' ')
        if len(line_) > 1:
            f2.write(line)
