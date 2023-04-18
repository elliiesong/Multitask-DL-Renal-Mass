import os
import SimpleITK as sitk
import shutil
import numpy as np


class_dict = {'I': 0, 'II': 1, 'IIF': 2, 'III': 3, 'IV': 4}


def get_dcm_files(path):
    files = []
    if not os.path.exists(path):
        return -1
    for filepath, dirs, names in os.walk(path):
        for filename in names:
            if filename.endswith('.dcm'):
                files.append(os.path.join(filepath, filename))
    return files


path = r'F:\NF 8.10 newest'
files = get_dcm_files(path)
multinid = []
for i in files:
    info = i.split('_')
    if len(info) == 4:
        multinid.append(info)
    else:
        print(i)
        dir_path = 'F:\\NF 8.10 concat\\' + info[0].split('\\')[-1]
        save_path = i.replace('NF 8.10 newest', 'NF 8.10 concat')
        os.makedirs(dir_path, exist_ok=True)
        shutil.copy(i, save_path)
        shutil.copy(i.replace('.dcm', '.nii'), save_path.replace('.dcm', '.nii'))

print('\n')
print('\n')
print('multinid process start')
for i in multinid:
    multinid.remove(i)
    for j in multinid:
        if i[0] == j[0] and i[2] == j[2]:
            print(i, j)
            multinid.remove(j)
            nii1 = i[0] + '_' + i[1] + '_' + i[2] + '_' + i[3].replace('.dcm', '.nii')
            nii2 = j[0] + '_' + j[1] + '_' + j[2] + '_' + j[3].replace('.dcm', '.nii')
            gt1 = sitk.GetArrayFromImage(sitk.ReadImage(nii1))
            gt2 = sitk.GetArrayFromImage(sitk.ReadImage(nii2))
            gt = gt1 + gt2
            gt[gt != 0] = 1
            nii = sitk.GetImageFromArray(gt)
            dcm = i[0] + '_' + i[1] + '_' + i[2] + '_' + i[3]
            label1 = class_dict[i[1]]
            label2 = class_dict[j[1]]
            if label1 >= label2:
                label = i[1]
            else:
                label = j[1]
            dir_path = 'C:\\Users\\admin\\Desktop\\NF3.19_concat\\' + info[0].split('\\')[-1]
            save_path = i[0] + '_' + label + '_' + i[2] + '.dcm'
            save_path = save_path.replace('NF3.19', 'NF3.19_concat')
            os.makedirs(dir_path, exist_ok=True)
            shutil.copy(dcm, save_path)
            sitk.WriteImage(nii, save_path.replace('.dcm', '.nii'))


