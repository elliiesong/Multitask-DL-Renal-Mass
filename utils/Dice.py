import SimpleITK as sitk
import numpy as np
import os


def get_nii_files(path):
    files = []
    if not os.path.exists(path):
        return -1
    for filepath, dirs, names in os.walk(path):
        for filename in names:
            if filename.endswith('.nii'):
                files.append(os.path.join(filepath, filename))
    return files


def Dice(y_true, y_pred):
    y_true = y_true
    y_pred = y_pred
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1)


path_gt = r'F:\nnUNetFrame\data\Testset\Testset3\gt'
path_pre = r'F:\nnUNetFrame\data\Testset\seg\seg\Testset3'
# path_gt = r'F:\nnUNetFrame\dataSeg\newDataGT\GY2023'
# path_pre = r'F:\nnUNetFrame\dataSeg\newDataSeg\GY202303Seg'
# path_gt = r"F:\renal_cyst\gt\NF2023"  #GT
# path_pre = r"F:\HAINANout\HAINANout"  #test
# path_pre = r'F:\renal_cyst\inferNew\inferTr'
txt_path = r"./"  #dice文档
# target = r'F:\renal_cyst\testdata'
# targetfiles = os.listdir(target)

dice_txt = open(os.path.join(txt_path, "dice.txt"), "r+")
dice_txt.truncate()    #每次运行该文件，对里面的数据清空

dices = []
pixel = []
# for file in os.listdir(path_pre):
#     if file.endswith('.nii.gz'):
#         label = file.split('_')[1]
#         # if label == 'I' or label == 'II' or label == 'IV':
#         #     continue
#
#         gt_array = np.load(os.path.join(path_gt, file.replace('.nii.gz', '.npy')))
#
#         pre_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(path_pre, file)))

for gt in os.listdir(path_gt):
    if gt.endswith('.npy'):
        info = gt.split('_')
        file = info[0] + '_' + info[1] + '_' + info[2]
        # file = info[0] + '_' + info[0] + '_' + info[1] + '_' + info[2]
        gt_array = np.load(os.path.join(path_gt, gt))
        try:
            pre_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(path_pre, file.replace('.npy', '.nii.gz'))))
        except:
            continue

        pixel.append(pre_array.sum())

        dice = Dice(gt_array, pre_array)
        if dice < 0.5:
            print(file)
        # if label == 'III':
        #     print(file, f'dice: {dice}')
        dices.append(dice)
        dice = str(dice)
        # dice_txt.write("Predict " + num + " :  ")
        dice_txt.write(dice)
        dice_txt.write("\n")

dice_txt.close()

print(f'the number of slices: {len(dices)}')

# count = 0
# for i in dices:
#     if i >= 0.5:
#         count += 1
dices = np.asarray(dices)

# print(f'the number of slices dice greader than 0.5: {count}')

# print(f'min dice: {dices.min()}')

# pixel = np.asarray(pixel)
print(f'the mean of dice: {dices.mean()}')

# print(f'min pixel: {pixel.min()} index:{pixel.argmin()} the dice: {dices[pixel.argmin()]}')
# print(f'mean pixel: {pixel.mean()}')

# a1 = 0
# a2 = 0
# a3 = 0
# a4 = 0
# a5 = 0
# a6 = 0
# for i in pixel:
#     if i < 100:
#         a1 += 1
#     elif i <500:
#         a2 += 1
#     elif i < 5000:
#         a3 += 1
#     elif i < 10000:
#         a4 += 1
#     elif i < 30000:
#         a5 += 1
#     else:
#         a6 += 1
# print(f'{a1} {a2} {a3} {a4} {a5} {a6}')


# print(gt_array.shape, gt_array.dtype)
# gt = gt.astype(np.uint8)
# print(gt.shape, gt.dtype)
