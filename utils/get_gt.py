import SimpleITK as sitk
import numpy as np
import os
import cv2
from skimage import measure

# 得到文件夹下所有的nii子文件路径的列表
def get_nii_files(path):
    files = []
    if not os.path.exists(path):
        return -1
    for filepath, dirs, names in os.walk(path):
        for filename in names:
            if filename.endswith('.nii'):
                files.append(os.path.join(filepath, filename))
    return files

# ct_path = r'C:\Users\admin\Desktop\MUMC UCSF mask'
# ct_path = r'D:\NF calcification'
# ct_path = r'F:\calcification'
# ct_path = r'F:\NF 8.10 concat'
# ct_path = r'F:\HN 8.10 newest'
# ct_path = r'F:\NF 7.20'
# ct_path = r'F:\ExValNF 11.19'
# ct_path = r'F:\NF 11.23'
ct_path = r'E:\Testset2(GY)'
# ct_path = r'F:\additional cases mask12.1'
# ct_path = r'E:\NFYY 2023.3 mask'
# ct_path = r'E:\GY mass 2023.3'
# ct_path = r'C:\Users\admin\Desktop\NF3.19_concat'
# ct_path = r'E:\5mm mask'
# ct_path = r'F:\HAINAN mask'
ct_files = get_nii_files(ct_path)
gt_path = r'F:\renal_cyst\gt\Testset2'
# gt_path = r'F:\nnUNetFrame\dataSeg\NF2023'
# gt_path = r'F:\nnUNetFrame\dataSeg\GY2023'
os.makedirs(gt_path, exist_ok=True)
for ct_image in ct_files:
    # Read niifile
    name = ct_image.split('\\')[-1].split('.')[0]
    info = name.split('_')
    if len(info[0].split('-')) == 2:
        info[0] = info[0].split('-')[0] + info[0].split('-')[1]
    newname = info[0] + '_' + info[1] + '_' + info[2]
    mask_array = sitk.GetArrayFromImage(sitk.ReadImage(ct_image)).squeeze()
    np.save(os.path.join(gt_path, newname + '.npy'), mask_array)
    print(name)
