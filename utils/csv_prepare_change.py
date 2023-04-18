import SimpleITK as sitk
import numpy as np
import os
import cv2
import pydicom
from skimage import measure
import matplotlib.pyplot as plt

# 得到文件夹下所有的nii子文件路径的列表
def get_nii_files(path):
    files = []
    if not os.path.exists(path):
        return -1
    for filepath, dirs, names in os.walk(path):
        for filename in names:
            if filename.endswith('.nii.gz'):
                files.append(os.path.join(filepath, filename))
    return files


def split_dataset(datalist, split_rate):
    num = len(datalist)
    train_num = int(num * split_rate)
    trainlist = datalist[:train_num]
    testlist = datalist[train_num:]
    return trainlist, testlist


train_path = r'D:\Wendy\renal_cyst\nnUNet\nnUNet_raw_data\Task200_kidney\imageTr_preprocessed'
test_path = r'D:\Wendy\renal_cyst\nnUNet\nnUNet_raw_data\Task200_kidney\imageTs_preprocessed'
train_files = get_nii_files(train_path)
test_files = get_nii_files(test_path)
images = {
    'train': [],
    'test': []
}
for file in train_files:
    train_info = file.split('\\')[-1].split('_')
    if len(train_info) == 3:
        images['train'].append(file)
for file in test_files:
    test_info = file.split('\\')[-1].split('_')
    if len(test_info) == 3:
        images['test'].append(file)

num = int(0.8 * len(images['train']))
train_files = images['train'][:num]
val_files = images['train'][num:]

save_path = r'/dataset'
test_path = r'/testdata'
png_path = r'/png'
png_train = os.path.join(png_path, 'train')
png_test = os.path.join(png_path, 'test')
os.makedirs(save_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)
os.makedirs(png_train, exist_ok=True)
os.makedirs(png_test, exist_ok=True)

train_labels = open(r'/dataset/train_labels.csv', 'w')
for ct_image in train_files:
    # Read niifile
    ct_img_array = sitk.GetArrayFromImage(sitk.ReadImage(ct_image)).squeeze()
    mask = ct_image.split('.')[0] + '_mask.nii.gz'
    mask_array = sitk.GetArrayFromImage(sitk.ReadImage(mask)).squeeze()

    if len(mask_array.shape) == 3:
        print(mask_array)
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
            np.save(save_full_path, ct_img_array.astype(np.float32))
            train_labels.write(save_name + ',' + str(x) + ',' + str(y) + ',' + str(x + w) + ',' + str(
                y + h) + ',' + class_name + '\n')

            plt.figure()
            plt.subplot(121)
            plt.title('image')
            plt.imshow(ct_img_array, cmap='gray')
            plt.subplot(122)
            plt.title('image * mask')
            plt.imshow(ct_img_array * mask_array, cmap='gray')
            plt.savefig(os.path.join(png_train, save_name.replace('.npy', '.png')))
            plt.close()

train_labels.close()


val_labels = open(r'F:\renal_cyst\dataset\val_labels.csv', 'w')
for ct_image in val_files:
    # Read niifile
    ct_img_array = sitk.GetArrayFromImage(sitk.ReadImage(ct_image)).squeeze()
    mask = ct_image.split('.')[0] + '_mask.nii.gz'
    mask_array = sitk.GetArrayFromImage(sitk.ReadImage(mask)).squeeze()

    if len(mask_array.shape) == 3:
        print(ct_image)
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
            np.save(save_full_path, ct_img_array.astype(np.float32))
            val_labels.write(save_name + ',' + str(x) + ',' + str(y) + ',' + str(x + w) + ',' + str(
                y + h) + ',' + class_name + '\n')

            plt.figure()
            plt.subplot(121)
            plt.title('image')
            plt.imshow(ct_img_array, cmap='gray')
            plt.subplot(122)
            plt.title('image * mask')
            plt.imshow(ct_img_array * mask_array, cmap='gray')
            plt.savefig(os.path.join(png_train, save_name.replace('.npy', '.png')))
            plt.close()

val_labels.close()


test_labels = open(r'F:\renal_cyst\testdata\test_labels.csv', 'w')
for ct_image in images['test']:
    # Read niifile
    ct_img_array = sitk.GetArrayFromImage(sitk.ReadImage(ct_image)).squeeze()
    mask = ct_image.split('.')[0] + '_mask.nii.gz'
    mask_array = sitk.GetArrayFromImage(sitk.ReadImage(mask)).squeeze()

    if len(mask_array.shape) == 3:
        print(mask_array)
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
            save_full_path = os.path.join(test_path, save_name)
            np.save(save_full_path, ct_img_array.astype(np.float32))
            test_labels.write(save_name + ',' + str(x) + ',' + str(y) + ',' + str(x + w) + ',' + str(
                y + h) + ',' + class_name + '\n')

            plt.figure()
            plt.subplot(121)
            plt.title('image')
            plt.imshow(ct_img_array, cmap='gray')
            plt.subplot(122)
            plt.title('image * mask')
            plt.imshow(ct_img_array * mask_array, cmap='gray')
            plt.savefig(os.path.join(png_test, save_name.replace('.npy', '.png')))
            plt.close()

test_labels.close()