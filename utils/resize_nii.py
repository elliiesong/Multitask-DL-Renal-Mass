import SimpleITK as sitk
import os
import numpy as np
from skimage.transform import resize
from collections import OrderedDict
from batchgenerators.augmentations.utils import resize_segmentation
from scipy.ndimage.interpolation import map_coordinates
import matplotlib.pyplot as plt
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian
import pydicom._storage_sopclass_uids


def resample_data_ski(data, new_shape, order=3):
    """
    separate_z=True will resample with order 0 along z
    :param data:
    :param new_shape:
    :param is_seg:
    :param axis:
    :param order:
    :param do_separate_z:
    :param cval:
    :param order_z: only applies if do_separate_z is True
    :return:
    """
    assert len(data.shape) == 3, "data must be (x, y, z)"
    kwargs = {'mode': 'edge', 'anti_aliasing': False}
    dtype_data = data.dtype
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if np.any(shape != new_shape):
        data = data.astype(float)
        reshaped_data = resize(data, new_shape, order, cval=0, **kwargs)
        return reshaped_data.astype(dtype_data)
    else:
        print("no resampling necessary")
        return data


def resample_patient(data, original_spacing, target_spacing, order=3):
    """
    :param cval_seg:
    :param cval_data:
    :param data:
    :param seg:
    :param original_spacing:
    :param target_spacing:
    :param order_data:
    :param order_seg:
    :param order_z_seg: only applies if do_separate_z is True
    :param order_z_data: only applies if do_separate_z is True
    :return:
    """
    shape = np.array(data.shape)
    print('shape', shape)
    print('original_spacing', original_spacing)
    print('target_spacing', target_spacing)
    size_rate = (np.array(original_spacing) / np.array(target_spacing)).astype(float)
    # print('size_rate', size_rate)
    size_rate = np.array([size_rate[2], size_rate[1], size_rate[0]])
    # print('size_rate', size_rate)
    new_shape = np.round((size_rate * shape)).astype(int)
    # print('new_shape', new_shape)
    data_reshaped = resample_data_ski(data, new_shape, order)
    print('data_reshaped', data_reshaped.shape)
    return data_reshaped


def write_dcm(data_array, ds, spacing, position, save_name, ImagePositionPatient):
    ds.WindowCenter = 40
    ds.WindowWidth = 400
    # print('ds.MediaStorageSOPInstanceUID', ds.file_meta.MediaStorageSOPInstanceUID)
    try:
        ds.file_meta.MediaStorageSOPInstanceUID = '.'.join(ds.file_meta.MediaStorageSOPInstanceUID.split('.')[:-1]) + \
                                        str(int(ds.file_meta.MediaStorageSOPInstanceUID.split('.')[-1]) + position)
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    except:
        pass
    # print('ds.MediaStorageSOPInstanceUID', ds.file_meta.MediaStorageSOPInstanceUID)
    ds.InstanceNumber = position
    ds.SliceLocation = ImagePositionPatient[2]
    ds.ImagePositionPatient = [ImagePositionPatient[0], ImagePositionPatient[1], ImagePositionPatient[2]+position]
    ds.SliceThickness = spacing[2]
    ds.Rows = data_array.shape[0]
    ds.Columns = data_array.shape[1]
    ds.PixelSpacing = [str(spacing[0]), str(spacing[1])]
    # ds.PhotometricInterpretation = "MONOCHROME2"
    # pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)
    print('ds', ds)
    print("Setting pixel data...")
    # plt.imshow(data_array, cmap='gray')
    # plt.show()
    # print('min', np.min(data_array), np.max(data_array))
    ds.PixelData = data_array.tobytes()
    ds.save_as(os.path.join(save_name, str(position) + '.dcm'))
    # ds = pydicom.dcmread(os.path.join(save_name, str(position) + '.dcm'))
    # print('ds.pixel_array', np.min(ds.pixel_array), np.max(ds.pixel_array))
    # # plt.imshow(data_array, cmap = 'gray')
    # # plt.show()
    # plt.imshow(ds.pixel_array, cmap='gray')
    # plt.show()

# path = r'D:\Wendy\normalized_3d\data_normalized'
# save_path = r'D:\Wendy\normalized_3d\data_resized'
# target_spacing = (0.68359375, 0.68359375, 5.0)
# files_path = [os.path.join(path, i) for i in os.listdir(path)]
# path = r'E:\foreign_data_for_normalization\2007-06__Studies\MUMC.087_MUMC.087_CT_2007-06-21_172809_CT.ABDOMEN_._n68__00000\2.16.840.1.114362.1.11956109.23089158482.564080929.246.5726.dcm'
# ds_sample = pydicom.dcmread(path)
# pixel_array = ds_sample.pixel_array
# pixel_array_stk = sitk.GetArrayFromImage(sitk.ReadImage(path))
# print('min, max', np.min(pixel_array), np.max(pixel_array))
# print('min, max', np.min(pixel_array_stk), np.max(pixel_array_stk))
#
# for file in files_path:
#     nii_img = sitk.ReadImage(file)
#     patient_name = file.split('\\')[-1]
#     print('patient_name', patient_name)
#     data = sitk.GetArrayFromImage(nii_img)
#     original_spacing = nii_img.GetSpacing()
#     origin = nii_img.GetOrigin()
#     # data = data.transpose((1, 2, 0))
#     print('data', data.shape, np.min(data[0, :, :]), np.max(data[0, :, :]))
#     data_new = resample_patient(data, original_spacing, target_spacing, order=3)
#     data_new[data_new < -120] = -120
#     data_new[data_new > 180] = 180
#     # plt.imshow(data_new[250, :, :], cmap='gray')
#     # plt.show()
#     # plt.imshow(data_new[:, :, 30], cmap='gray')
#     # plt.show()
#     #
#     print('data_new', data_new.shape)
#
#     # out = sitk.GetImageFromArray(data_new)
#     # out.SetOrigin(origin)
#     # out.SetSpacing(target_spacing)
#     # result_name_path = os.path.join(save_path, patient_name + '.nii.gz')
#     # # print('result_name_path', result_name_path)
#     # sitk.WriteImage(out, result_name_path)
#     dcm_path = r'D:\Wendy\normalized_3d\dcm_path'
#     pat_dcm_path = os.path.join(dcm_path, patient_name)
#     if not os.path.exists(pat_dcm_path):
#         os.mkdir(pat_dcm_path)
#     count = 0
#     for idx in range(data_new.shape[0]):
#         count += 1
#         write_dcm(data_new[idx], ds_sample, target_spacing, count, pat_dcm_path)

