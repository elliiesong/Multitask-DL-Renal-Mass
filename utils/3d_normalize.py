import SimpleITK as sitk
import os
import pydicom
import matplotlib.pyplot as plt
import numpy as np
from resize_nii import resample_patient, write_dcm


def sort_posi(img_npy_list, img_posi_list):
    sorted_id = sorted(range(len(img_posi_list)), key=lambda k: img_posi_list[k], reverse=False)
    img_npy_list_sort = []
    for q in range(len(sorted_id)):
        img_npy_list_sort.append(img_npy_list[sorted_id[q]])
    img_npy_list_sort = np.array(img_npy_list_sort)
    # print('train_patient_path_sort', train_patient_path_sort)
    # print('img_npy_list_sort.shape', img_npy_list_sort.shape)
    return img_npy_list_sort, sorted_id


def read_dcm(patient_path):
    slices_path = os.listdir(patient_path)
    slices_npy = []
    img_posi_list = []
    for slice in slices_path:
        slice_dicom = pydicom.read_file(os.path.join(patient_path, slice))
        # print('slice_dicom', slice_dicom.SliceLocation, slice_dicom.ImagePositionPatient)
        # print('slice_dicom.MediaStorageSOPInstanceUID', slice_dicom.file_meta.MediaStorageSOPInstanceUID)
        # print(slice_dicom)
        spacing = slice_dicom.PixelSpacing
        thickness = slice_dicom.SliceThickness
        slice_npy = slice_dicom.pixel_array
        slices_npy.append(slice_npy)
        img_posi_list.append(slice_dicom.ImagePositionPatient[-1])
    slices_npy_sort, sorted_id = sort_posi(slices_npy, img_posi_list)
    # plt.imshow(slices_npy_sort[:, :, 250], cmap='gray')
    # plt.show()
    return slices_npy_sort, sorted_id, spacing, thickness, slice_dicom


def resample_process_3d(pat_path, patient_name):
    target_spacing = (0.68359375, 0.68359375, 5.0)
    image_array_order, sorted_id, spacing, thickness, ds_sample = read_dcm(pat_path)
    original_spacing = (spacing[0], spacing[1], thickness)
    print('pat_path', pat_path)
    slice_names = [os.path.join(pat_path, i) for i in os.listdir(pat_path)]
    slice_names_new = []
    for q in range(len(sorted_id)):
        slice_names_new.append(slice_names[sorted_id[q]])
    data_new = resample_patient(image_array_order, original_spacing, target_spacing, order=3)
    dcm_path = r'E:\normalized data2023.3'
    pat_dcm_path = os.path.join(dcm_path, patient_name)
    print('pat_dcm_path', pat_dcm_path)
    if not os.path.exists(pat_dcm_path):
        os.makedirs(pat_dcm_path)
    count = 0
    for idx in range(data_new.shape[0]):
        count += 1
        write_dcm(data_new[idx], ds_sample, target_spacing, count, pat_dcm_path, ds_sample.ImagePositionPatient)


# path = r'F:\外部验证补充I类（薄层）'
path = r'E:\unnormalized data2023.3'
pats = os.listdir(path)
for pat in pats:
    if pat not in ['NFYY01']:
        continue
    print('pat', pat)
    pat_path = os.path.join(path, pat)
    pat_path_end = pat_path
    pat_list = os.listdir(pat_path)
    if len(pat_list) < 3:
        pat_path_end = os.path.join(pat_path, pat_list[0])
        if len(os.listdir(pat_path_end)) == 1:
            pat_path_end = os.path.join(pat_path_end, os.listdir(pat_path_end)[0])
        print('pat_path_end', pat_path_end, os.listdir(pat_path_end))
    resample_process_3d(pat_path_end, pat)
