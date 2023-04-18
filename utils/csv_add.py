import SimpleITK as sitk
import numpy as np
import os
import cv2
import pandas as pd
import nibabel as nib
import nibabel.processing as nibproc
from skimage import measure
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom

# 得到文件夹下所有的dcm子文件路径的列表
def get_dcm_files(path):
    files = []
    if not os.path.exists(path):
        return -1
    for filepath, dirs, names in os.walk(path):
        for filename in names:
            if filename.endswith('.dcm') or filename.split('.')[-1] != 'nii':
            # if filename.endswith('.dcm'):
                files.append(os.path.join(filepath, filename))
    return files


def show_img_info(itk_img):
    origin = np.array(itk_img.GetOrigin())
    print('Origin (x, y, z): ', origin)

    direction = np.array(itk_img.GetDirection())
    print('direction: ', direction)

    spacing = np.array(itk_img.GetSpacing())
    print('Spacing (x, y, z): ', spacing)


def projectImage(target_image, input_image):
    print('target_image', target_image)
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(target_image)
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    resampled_img = resample.Execute(input_image)
    return resampled_img


def split_dataset(datalist, split_rate):
    num = len(datalist)
    if type(split_rate) == list:
        [sr1, sr2] = split_rate
        train_num = int(num * sr1)
        val_num = int(num * sr2)
        trainlist = datalist[:train_num]
        vallist = datalist[train_num:val_num]
        testlist = datalist[val_num:]
        return trainlist, vallist, testlist
    elif type(split_rate) == float:
        train_num = int(num * split_rate)
        trainlist = datalist[:train_num]
        vallist = datalist[train_num:]
        return trainlist, vallist
    else:
        raise ValueError(f'the type of split_rate is Error, got {type(split_rate)}')


def resampling_reorientation(img, out_shape, voxel_size):
    resampleIm = nibproc.conform(img, out_shape=out_shape, voxel_size=voxel_size, order=1, cval=0.0)
    return resampleIm


def dicom_to_nii(path):
    ct_img = sitk.ReadImage(path)
    spacing = ct_img.GetSpacing()
    ct_img_array = sitk.GetArrayFromImage(ct_img).squeeze()
    affine = np.diag((-1, -1, 1, 1))
    ct_img_nii = nib.Nifti1Image(ct_img_array, affine)
    return ct_img_nii, spacing, ct_img_array.shape


# ct_path = r'E:\5mm mask'
# ct_path = r'D:\NF calcification'
# ct_path = r'F:\calcification'
# ct_path = r'F:\HAINAN mask'
# ct_path = r'F:\HN 8.10 newest'
# ct_path = r'F:\NF 7.20'
# ct_path = r'F:\ExValNF 11.19'
# ct_path = r'F:\NF 11.23'
ct_path = r'E:\Testset2(GY)'
# ct_path = r'F:\additional cases mask12.1'
# ct_path = r'E:\GY mass 2023.3'
# ct_path = r'E:\NFYY 2023.3 mask'
ct_files = get_dcm_files(ct_path)
test_files = ct_files
# train_files, val_files = split_dataset(ct_files, 0.8)
save_path = r'F:\renal_cyst\dataset'
test_path = r'F:\renal_cyst\testdata\Testset2'
os.makedirs(save_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# 额外加train
# df = pd.read_csv(r'F:\renal_cyst\dataset\train_labels.csv')
# i = len(df)
# for ct_image in train_files:
#     ct_img_array = sitk.GetArrayFromImage(sitk.ReadImage(ct_image)).squeeze()
#     mask_array = sitk.GetArrayFromImage(sitk.ReadImage(ct_image.replace('.dcm', '.nii'))).squeeze()
#     # ct_img_array = zoom(ct_img_array, (512 / ct_img_array.shape[0], 512 / ct_img_array.shape[1]))
#     # mask_array = zoom(mask_array, (512 / mask_array.shape[0], 512 / mask_array.shape[1]))
#
#     if len(mask_array.shape) == 3:
#         print(mask_array)
#         continue
#
#     file_name = os.path.splitext(ct_image.split('\\')[-1])[0]
#     num_name = file_name.split('_')[0]
#     class_name = file_name.split('_')[1]
#     slice = file_name.split('_')[-1]
#     roi = mask_array
#     y_pix, x_pix = np.where(roi == 1)
#     if len(x_pix) > 0 and len(measure.find_contours(roi, 0.5)) > 0:
#         print(ct_image)
#         roi = mask_array.astype(np.uint8)
#         contours, hierarchy = cv2.findContours(roi, 1, 2)
#         for cnt_index in range(len(contours)):
#             cnt = contours[cnt_index]
#             x, y, w, h = cv2.boundingRect(cnt)
#             save_name = num_name + '_' + class_name + '_' + slice + '.npy'
#             save_full_path = os.path.join(save_path, save_name)
#             np.save(save_full_path, ct_img_array.astype(np.int16))
#             df.loc[i] = [save_name, str(x), str(y), str(x + w), str(y + h), class_name]
#             i += 1
# df.to_csv(r'F:\renal_cyst\dataset\train_labels.csv', index=None)

# 额外加val
# df = pd.read_csv(r'F:\renal_cyst\dataset\val_labels.csv')
# i = len(df)
# for ct_image in val_files:
#     ct_img_array = sitk.GetArrayFromImage(sitk.ReadImage(ct_image)).squeeze()
#     mask_array = sitk.GetArrayFromImage(sitk.ReadImage(ct_image.replace('.dcm', '.nii'))).squeeze()
#     # ct_img_array = zoom(ct_img_array, (512 / ct_img_array.shape[0], 512 / ct_img_array.shape[1]))
#     # mask_array = zoom(mask_array, (512 / mask_array.shape[0], 512 / mask_array.shape[1]))
#
#     if len(mask_array.shape) == 3:
#         print(ct_image)
#         continue
#
#     file_name = os.path.splitext(ct_image.split('\\')[-1])[0]
#     num_name = file_name.split('_')[0]
#     class_name = file_name.split('_')[1]
#     slice = file_name.split('_')[-1]
#     roi = mask_array
#     y_pix, x_pix = np.where(roi == 1)
#     if len(x_pix) > 0 and len(measure.find_contours(roi, 0.5)) > 0:
#         print(ct_image)
#         roi = mask_array.astype(np.uint8)
#         contours, hierarchy = cv2.findContours(roi, 1, 2)
#         for cnt_index in range(len(contours)):
#             cnt = contours[cnt_index]
#             x, y, w, h = cv2.boundingRect(cnt)
#             save_name = num_name + '_' + class_name + '_' + slice + '.npy'
#             save_full_path = os.path.join(save_path, save_name)
#             np.save(save_full_path, ct_img_array.astype(np.int16))
#             df.loc[i] = [save_name, str(x), str(y), str(x + w), str(y + h), class_name]
#             i += 1
# df.to_csv(r'F:\renal_cyst\dataset\val_labels.csv', index=None)

# 额外加test
# df = pd.read_csv(os.path.join(test_path, 'test_labels.csv'))
# i = len(df)
# for ct_image in test_files:
#     # Read niifile
#     ct_img_array = sitk.GetArrayFromImage(sitk.ReadImage(ct_image)).squeeze()
#     mask_array = sitk.GetArrayFromImage(sitk.ReadImage(ct_image.replace('.dcm', '.nii'))).squeeze()
#     # ct_img_array = zoom(ct_img_array, (512 / ct_img_array.shape[0], 512 / ct_img_array.shape[1]))
#     # mask_array = zoom(mask_array, (512 / mask_array.shape[0], 512 / mask_array.shape[1]))
#
#     if len(mask_array.shape) == 3:
#         print(mask_array)
#         continue
#     print(ct_image)
#
#     file_name = os.path.splitext(ct_image.split('\\')[-1])[0]
#     num_name = file_name.split('_')[0]
#     class_name = file_name.split('_')[1]
#     slice = file_name.split('_')[-1]
#     roi = mask_array
#     y_pix, x_pix = np.where(roi == 1)
#     if len(x_pix) > 0 and len(measure.find_contours(roi, 0.5)) > 0:
#         roi = mask_array.astype(np.uint8)
#         contours, hierarchy = cv2.findContours(roi, 1, 2)
#         for cnt_index in range(len(contours)):
#             cnt = contours[cnt_index]
#             x, y, w, h = cv2.boundingRect(cnt)
#             save_name = num_name + '_' + class_name + '_' + slice + '.npy'
#             save_full_path = os.path.join(test_path, save_name)
#             np.save(save_full_path, ct_img_array.astype(np.int16))
#             df.loc[i] = [save_name, str(x), str(y), str(x + w), str(y + h), class_name]
#             i += 1
# df.to_csv(os.path.join(test_path, 'test_labels.csv'), index=None)

# 重新加test
test_labels = open(os.path.join(test_path, 'test_labels.csv'), 'w')
for ct_image in test_files:
    # Read niifile
    ct_img_array = sitk.GetArrayFromImage(sitk.ReadImage(ct_image)).squeeze()
    if ct_image.endswith('.dcm'):
        ct_mask = ct_image.replace('.dcm', '.nii')
    elif ct_image.split('.')[-1] != 'nii':
        ct_mask = ct_image + '.nii'
    mask_array = sitk.GetArrayFromImage(sitk.ReadImage(ct_mask)).squeeze()
    # ct_img_array = zoom(ct_img_array, (512 / ct_img_array.shape[0], 512 / ct_img_array.shape[1]))
    # mask_array = zoom(mask_array, (512 / mask_array.shape[0], 512 / mask_array.shape[1]))

    if len(mask_array.shape) == 3:
        print(mask_array)
        continue
    print(ct_image)

    file_name = os.path.splitext(ct_image.split('\\')[-1])[0]
    num_name = file_name.split('_')[0]
    if len(num_name.split('-')) == 2:
        num_name = num_name.split('-')[0] + num_name.split('-')[1]
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
            np.save(save_full_path, ct_img_array.astype(np.int16))
            test_labels.write(save_name + ',' + str(x) + ',' + str(y) + ',' + str(x + w) + ',' + str(
                y + h) + ',' + class_name + '\n')
test_labels.close()