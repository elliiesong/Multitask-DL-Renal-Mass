import os
import SimpleITK as sitk
import numpy as np
from shutil import copyfile


def get_files(path):
    files = []
    if not os.path.exists(path):
        return -1
    for filepath, dirs, names in os.walk(path):
        for filename in names:
            if filename.endswith('.dcm') or filename.split('.')[-1] != 'nii':
                # if filename.endswith('.dcm'):
                # niifile = filename.replace('.dcm', '.nii')
                files.append(os.path.join(filepath, filename))
    return files


# concatDict1 = {
#     'GY28_II_6': 'GY28_I_7',
#     'GY37_IIF_9': 'GY37_II_10',
#     'GY38_I_4': 'GY38_I_5',
#     'GY38_II_9': 'GY38_I_10',
#     'GY38_II_7': 'GY38_I_8',
#     'GY41_I_10': 'GY41_II_9',
#     'GY41_I_8': 'GY41_II_7',
#     'GY41_I_6': 'GY41_II_5',
#     'GY48_I_3': 'GY48_III_4',
#     'GY63_III_6': 'GY63_IV_5',
#     'GY63_IV_7': 'GY63_IIF_8',
#     'GY64_I_6': 'GY64_IV_5',
#     'GY72_II_4': 'GY72_IV_3',
# }
concatDict1 = {}

concatDict2 = dict([val, key] for key, val in concatDict1.items())

if __name__ == "__main__":
    origin = r'E:\Testset3(NF)'
    out_dir = r'F:\nnUNetFrame\data\Testset\Testset3'
    img_dir = os.path.join(out_dir, 'img')
    gt_dir = os.path.join(out_dir, 'gt')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    files = get_files(origin)

    for i in range(len(files)):
        dcm1 = files[i]
        _dir = os.path.dirname(dcm1)
        fname = files[i].split('\\')[-1].replace('.dcm', '')
        if fname in concatDict1:
            fname2 = concatDict1[fname]
        elif fname in concatDict2:
            fname2 = concatDict2[fname]
        else:
            fname2 = ''

        if fname2 != '':
            dcm2 = os.path.join(_dir, fname2 + '.dcm')
            nii1 = dcm1.replace('.dcm', '.nii')
            nii2 = dcm2.replace('.dcm', '.nii')
            gt1 = sitk.GetArrayFromImage(sitk.ReadImage(nii1))
            gt2 = sitk.GetArrayFromImage(sitk.ReadImage(nii2))
            gt = gt1 + gt2
            gt[gt != 0] = 1
        else:
            nii1 = dcm1.replace('.dcm', '.nii')
            gt = sitk.GetArrayFromImage(sitk.ReadImage(nii1))

        np.save(os.path.join(gt_dir, fname + '.npy'), gt)

        # fname = files[i].split('\\')[-2] + '_' + files[i].split('\\')[-1]
        # if len(fname.split('_')) == 2:
        #     fname = fname.split('_')[0] + fname.split('_')[1]
        fname = fname + '_0000.nii.gz'
        # fname = fname + '_0000.nii.gz'
        print(fname)
        img = sitk.ReadImage(dcm1)
        sitk.WriteImage(img, os.path.join(img_dir, fname))
