import SimpleITK as sitk
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import cv2
import pydicom
import nibabel as nib
from skimage import measure

# 得到文件夹下所有的nii子文件路径的列表
def get_dcm_files(path):
    files = []
    if not os.path.exists(path):
        return -1
    for filepath, dirs, names in os.walk(path):
        for filename in names:
            if filename.endswith('.dcm'):
                files.append(os.path.join(filepath, filename))
    return files


def split_dataset(datalist, split_rate):
    num = len(datalist)
    train_num = int(num * split_rate)
    trainlist = datalist[:train_num]
    testlist = datalist[train_num:]
    return trainlist, testlist


def findslice(path):
    files = []
    for file in os.listdir(path):
        files.append(file)
    return files

train_path = 'F:/renal_cyst/trainingData/detection/coco/train2017'
val_path = 'F:/renal_cyst/trainingData/detection/coco/val2017'
train_names = findslice(train_path)
val_names = findslice(val_path)
ct_path = r'C:/Users/admin/Desktop/NF segmentation'
ct_files = get_dcm_files(ct_path)
save_path = r'/dataset'

# train_labels = open(r'F:\renal_cyst\dataset\train_labels.csv', 'w')
# for ct_image in ct_files:
#     # Read niifile
#     if ct_image.split('\\')[-1].replace('.dcm', '.npy') in train_names:
#         ct_img_array = sitk.GetArrayFromImage(sitk.ReadImage(ct_image)).squeeze()
#         mask_array = sitk.GetArrayFromImage(sitk.ReadImage(ct_image.replace('.dcm', '.nii'))).squeeze()
#         if len(mask_array.shape) == 3:
#             continue
#         print(ct_image)
#
#         file_name = os.path.splitext(ct_image.split('\\')[-1])[0]
#         num_name = file_name.split('_')[0]
#         class_name = file_name.split('_')[1]
#         slice = file_name.split('_')[-1]
#         roi = mask_array
#         y_pix, x_pix = np.where(roi == 1)
#         if len(x_pix) > 0 and len(measure.find_contours(roi, 0.5)) > 0:
#             roi = mask_array.astype(np.uint8)
#             contours, hierarchy = cv2.findContours(roi, 1, 2)
#             for cnt_index in range(len(contours)):
#                 cnt = contours[cnt_index]
#                 x, y, w, h = cv2.boundingRect(cnt)
#                 save_name = num_name + '_' + class_name + '_' + slice + '.npy'
#                 save_full_path = os.path.join(save_path, save_name)
#                 # np.save(save_full_path, ct_img_array.astype(np.int16))
#                 train_labels.write(save_name + ',' + str(x) + ',' + str(y) + ',' + str(x + w) + ',' + str(
#                     y + h) + ',' + class_name + '\n')
# train_labels.close()


val_labels = open(r'F:\renal_cyst\dataset\test_labels.csv', 'w')
for ct_image in ct_files:
    # Read niifile
    if ct_image.split('\\')[-1].replace('.dcm', '.npy') in val_names:
        ct_img_array = sitk.GetArrayFromImage(sitk.ReadImage(ct_image)).squeeze()
        mask_array = sitk.GetArrayFromImage(sitk.ReadImage(ct_image.replace('.dcm', '.nii'))).squeeze()
        if len(mask_array.shape) == 3:
            continue
        print(ct_image)

        file_name = os.path.splitext(ct_image.split('\\')[-1])[0]
        num_name = file_name.split('_')[0]
        class_name = file_name.split('_')[1]
        slice = file_name.split('_')[-1]
        roi = mask_array
        y_pix, x_pix = np.where(roi == 1)
        if len(x_pix) > 0 and len(measure.find_contours(roi, 0.5)) > 0:
            roi = mask_array.astype(np.uint8)
            contours, hierarchy = cv2.findContours(roi, 1, 2)
            for cnt_index in range(len(contours)):
                cnt = contours[cnt_index]
                x, y, w, h = cv2.boundingRect(cnt)
                save_name = num_name + '_' + class_name + '_' + slice + '.npy'
                save_full_path = os.path.join(save_path, save_name)
                # np.save(save_full_path, ct_img_array.astype(np.int16))
                val_labels.write(save_name + ',' + str(x) + ',' + str(y) + ',' + str(x + w) + ',' + str(
                    y + h) + ',' + class_name + '\n')
val_labels.close()

