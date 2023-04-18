import os
import pydicom
import scipy.stats as st
import numpy as np


# 得到文件夹下所有的nii子文件路径的列表
def get_nii_files(path):
    files = []
    if not os.path.exists(path):
        return -1
    for filepath, dirs, names in os.walk(path):
        for filename in names:
            # if filename.endswith('.dcm') or filename.split('.')[-1] != 'nii':
            if filename.endswith('.nii'):
                files.append(os.path.join(filepath, filename))
    return files


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


def getInterval(x, alpha=0.95):
    x_mean = np.mean(x)
    x_std = np.std(x)
    x_n = len(x)
    x_se = x_std / np.sqrt(x_n)
    x_ci = st.t.interval(alpha, x_n - 1, loc=x_mean, scale=x_se)
    return x_ci


dataPath = {
    'Development': {
        'imgPath': r'C:\Users\admin\Desktop\NF3.19_concat',
        'gtPath': r'F:\renal_cyst\gt\native',
    },
    'hainan': {
        'imgPath': r'F:\HN 8.10 newest',
        'gtPath': r'F:\renal_cyst\gt\HAINAN',
    },
    'foreign': {
        'imgPath': [r'E:\5mm mask', r'F:\additional cases mask12.1'],
        'gtPath': r'F:\renal_cyst\gt\foreign',
    },
    'nf': {
        'imgPath': [r'F:\NF 11.23', r'E:\NFYY 2023.3 mask'],
        'gtPath': r'F:\renal_cyst\gt\NFmixed',
    },
    'gy': {
        'imgPath': r'E:\GY mass 2023.3',
        'gtPath': r'F:\renal_cyst\gt\GY mass 2023.3'
    },
    'testset2': {
        'imgPath': r'E:\Testset2(GY)',
        'gtPath': r'F:\nnUNetFrame\data\Testset\Testset2\gt',
    },
    'testset3': {
        'imgPath': r'E:\Testset3(NF)',
        'gtPath': r'F:\nnUNetFrame\data\Testset\Testset3\gt'
    }
}


if __name__ == '__main__':
    dataset = 'testset3'
    if dataset in ['foreign', 'nf']:
        imgFiles = get_dcm_files(dataPath[dataset]['imgPath'][0]) + get_dcm_files(dataPath[dataset]['imgPath'][1])
    else:
        imgFiles = get_dcm_files(dataPath[dataset]['imgPath'])
    area = np.zeros(len(imgFiles))
    for i, file in enumerate(imgFiles):
        dcmData = pydicom.read_file(file)
        spacing = dcmData.PixelSpacing
        areaPerPixel = spacing[0] * spacing[1]
        if dataset == 'hainan':
            gtFile = os.path.join(dataPath[dataset]['gtPath'], file.split('\\')[-1] + '.npy')
        elif dataset == 'foreign':
            gtFile = os.path.join(dataPath[dataset]['gtPath'], file.split('\\')[-1].replace('-', '').replace('.dcm', '.npy'))
        else:
            gtFile = os.path.join(dataPath[dataset]['gtPath'], file.split('\\')[-1].replace('.dcm', '.npy'))
        pixelNum = np.load(gtFile).sum()
        area[i] = pixelNum * areaPerPixel
        print(area[i])
    print(f'{dataset} mean area: {area.mean()}, (95% confidence interval: {getInterval(area)})')