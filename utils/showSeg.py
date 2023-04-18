import os
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt
import matplotlib as mpl


if __name__ == '__main__':
    # 国外数据
    # imgDir = r'F:\foreignAll'
    # preDir = r'F:\renal_cyst\inferences\foreignAllout'
    # gtDir = r'F:\renal_cyst\gt\foreign'
    # outDir = r'F:\renal_cyst\segResult\foreign'

    # 海南数据
    # imgDir = r'F:\HNseg'
    # preDir = r'F:\HAINANout\HAINANout'
    # gtDir = r'F:\renal_cyst\gt\HAINAN'
    # outDir = r'F:\renal_cyst\segResult\hainan'

    # 南方外部验证数据
    # imgDir = r'F:\NF1123seg'
    # preDir = r'F:\renal_cyst\inferences\NF1123segout'
    # gtDir = r'F:\renal_cyst\gt\NF 11.23'
    # outDir = r'F:\renal_cyst\segResult\NF 11.23'

    # 南方新增外部验证数据
    # imgDir = r'F:\renal_cyst\testdata\NFYY 2023.3 mask'
    # preDir = r'F:\nnUNetFrame\dataSeg\newDataSeg\NF202303Seg'
    # gtDir = r'F:\renal_cyst\gt\NFYY 2023.3 mask'
    # outDir = r'F:\renal_cyst\segResult\NFYY 2023.3 mask'

    # 广医外部验证数据
    imgDir = r'F:\renal_cyst\testdata\GY mass 2023.3'
    preDir = r'F:\nnUNetFrame\dataSeg\newDataSeg\GY202303Seg'
    gtDir = r'F:\renal_cyst\gt\GY mass 2023.3'
    outDir = r'F:\renal_cyst\segResult\GY mass 2023.3'


    # 内部验证
    # imgDir = r'F:\renal_cyst\dataset'
    # preDir = r'F:\renal_cyst\inferTr'
    # gtDir = r'F:\renal_cyst\gt\native'
    # outDir = r'F:\renal_cyst\segResult\val'
    os.makedirs(outDir, exist_ok=True)

    for file in os.listdir(gtDir):
        # for NF 11.23
        # gt_file = file
        # file = file.split('_')[0] + '_' + file

        # for HN
        # pre_file = file.split('_')[0] + '_' + file.split('_')[2]

        # for val

        imgPath = os.path.join(imgDir, file)
        img_array = np.load(imgPath)
        # img_array = sitk.GetArrayFromImage(sitk.ReadImage(imgPath)).squeeze()
        gt_array = np.load(os.path.join(gtDir, file))
        try:
            pre_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(preDir, file.replace('.npy', '.nii.gz')))).squeeze()
        except:
            print(file)
            continue

        img_array[img_array < -20] = -20
        img_array[img_array > 500] = 500
        img_array = (img_array + 20) / 520 * 255
        img_array = img_array.astype('uint8')

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(img_array)
        plt.title('Original CT')
        plt.xticks(())
        plt.yticks(())

        imgShow = img_array.copy()
        img_array[gt_array == 1] = 200
        gtShow = img_array
        img_array = imgShow
        img_array[pre_array == 1] = 255
        preShow = img_array

        plt.subplot(1, 3, 2)
        plt.imshow(gtShow)
        plt.title('Manually segmented')
        plt.xticks(())
        plt.yticks(())
        plt.subplot(1, 3, 3)
        plt.imshow(preShow)
        plt.title('AI segmented')
        plt.xticks(())
        plt.yticks(())
        # print(file)
        plt.savefig(os.path.join(outDir, file.replace('.npy', '.png')))
        plt.close()