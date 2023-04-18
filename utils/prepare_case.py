import os
import json
import numpy as np
import pandas as pd
import glob
import cv2
import shutil
from IPython import embed
from sklearn.model_selection import train_test_split

import SimpleITK as sitk
import random
import matplotlib.pyplot as plt
import cv2
from skimage import measure

#0为背景
classname_to_id = {"infarction": 1}

class Csv2CoCo:
    def __init__(self,image_dir,total_annos):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.image_dir = image_dir
        self.total_annos = total_annos

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w'), ensure_ascii=False, indent=2)  # indent=2 更加美观显示

    # 由txt文件构建COCO
    def to_coco(self, keys):
        self._init_categories()
        for key in keys:
            self.images.append(self._image(key))
            shapes = self.total_annos[key]
            for shape in shapes:
                bboxi = []
                for cor in shape[:-1]:
                    bboxi.append(int(cor))
                label = shape[-1]
                annotation = self._annotation(bboxi,label)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'DCM dataset'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # 构建类别
    def _init_categories(self):
        for k, v in classname_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    # 构建COCO的image字段
    def _image(self, path):
        image = {}
        print(path)
        img = np.load(os.path.join(self.image_dir, path))
        image['height'] = img.shape[0]
        image['width'] = img.shape[1]
        image['id'] = self.img_id
        image['file_name'] = path
        return image

    # 构建COCO的annotation字段
    def _annotation(self, shape,label):
        # label = shape[-1]
        points = shape[:4]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        #annotation['category_id'] = int(classname_to_id[label])
        annotation['category_id'] = 1
        annotation['segmentation'] = self._get_seg(points)
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = self._get_area(points)
        return annotation

    # COCO的格式： [x1,y1,w,h] 对应COCO的bbox格式
    def _get_box(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        return [min_x, min_y, max_x - min_x, max_y - min_y]
    # 计算面积
    def _get_area(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        return (max_x - min_x+1) * (max_y - min_y+1)
    # segmentation
    def _get_seg(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        h = max_y - min_y
        w = max_x - min_x
        a = []
        a.append([min_x,min_y, min_x,min_y+0.5*h, min_x,max_y, min_x+0.5*w,max_y, max_x,max_y, max_x,max_y-0.5*h, max_x,min_y, max_x-0.5*w,min_y])
        return a


train_csv_file = r"F:\renal_cyst\HAINAN\train_labels.csv"
test_csv_file = r"F:\renal_cyst\HAINAN\test_labels.csv"
image_dir = r"/HAINAN"
saved_coco_path = "../trainingData/detection"
# 整合csv格式标注文件
train_csv_annotations = {}
annotations = pd.read_csv(train_csv_file, header=None).values
for annotation in annotations:
    key = annotation[0].split(os.sep)[-1]
    value = np.array([annotation[1:]])
    if key in train_csv_annotations.keys():
        train_csv_annotations[key] = np.concatenate((train_csv_annotations[key], value), axis=0)
    else:
        train_csv_annotations[key] = value

test_csv_annotations = {}
annotations = pd.read_csv(test_csv_file, header=None).values
for annotation in annotations:
    key = annotation[0].split(os.sep)[-1]
    value = np.array([annotation[1:]])
    if key in test_csv_annotations.keys():
        test_csv_annotations[key] = np.concatenate((test_csv_annotations[key], value), axis=0)
    else:
        test_csv_annotations[key] = value

# 按照键值划分数据
train_keys = list(train_csv_annotations.keys())
val_keys = list(test_csv_annotations.keys())

print("train_n:", len(train_keys), 'val_n:', len(val_keys))
# 创建必须的文件夹
if not os.path.exists('%s/coco/annotations/' % saved_coco_path):
    os.makedirs('%s/coco/annotations/' % saved_coco_path)
if not os.path.exists('%scoco/train2017/' % saved_coco_path):
    os.makedirs('%s/coco/train2017/' % saved_coco_path)
if not os.path.exists('%s/coco/val2017/' % saved_coco_path):
    os.makedirs('%s/coco/val2017/' % saved_coco_path)
# 把训练集转化为COCO的json格式
l2c_train = Csv2CoCo(image_dir=image_dir, total_annos=train_csv_annotations)
train_instance = l2c_train.to_coco(train_keys)
l2c_train.save_coco_json(train_instance, '%s/coco/annotations/instances_train2017.json' % saved_coco_path)
for file in train_keys:
    shutil.copy(os.path.join(image_dir, file), "%s/coco/train2017/" % saved_coco_path)
for file in val_keys:
    shutil.copy(os.path.join(image_dir, file), "%s/coco/val2017/" % saved_coco_path)
# 把验证集转化为COCO的json格式
l2c_val = Csv2CoCo(image_dir=image_dir, total_annos=test_csv_annotations)
val_instance = l2c_val.to_coco(val_keys)
l2c_val.save_coco_json(val_instance, '%s/coco/annotations/instances_val2017.json' % saved_coco_path)