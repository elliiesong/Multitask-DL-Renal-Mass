import SimpleITK as sitk
import numpy as np
import os


# 得到文件夹下所有的nii子文件路径的列表
def get_dcm_files(path):
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


split_rate = 0.8
ct_path = r'/inferTs'
ct_files = get_dcm_files(ct_path)
train_files, test_files = split_dataset(ct_files, split_rate)
save_path = r'/infer'

print('traindata')
for image in train_files:
    array = sitk.GetArrayFromImage(sitk.ReadImage(image)).squeeze()
    img_name = image.split('\\')[-1].split('.')[0]
    print(img_name)
    np.save(os.path.join(save_path + '\\train', img_name), array.astype(np.int16))

print('testdata')
for image in test_files:
    array = sitk.GetArrayFromImage(sitk.ReadImage(image)).squeeze()
    img_name = image.split('\\')[-1].split('.')[0]
    print(img_name)
    np.save(os.path.join(save_path + '\\test', img_name), array.astype(np.int16))