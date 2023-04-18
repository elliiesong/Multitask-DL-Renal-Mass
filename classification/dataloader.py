import pandas as pd
import pydicom
from torchvision import transforms, datasets
import os
import torch
from torch.utils.data import WeightedRandomSampler
from PIL import Image
from scipy.ndimage.interpolation import zoom
import SimpleITK as sitk
import numpy as np
from skimage import measure
import torchvision
import torch
from torch.autograd import Variable
import cv2
import re
import matplotlib.pyplot as plt
from scipy.ndimage.filters import median_filter
from scipy.signal import convolve2d
from skimage.transform import resize
from batchgenerators.augmentations.utils import resize_segmentation
from augmentation import augmentation_func
import random
from fold import Kfold, getRandom
import operator
import functools


class_dict = {'I': 0, 'II': 1, 'IIF': 2, 'III': 3, 'IV': 4}

class_dict_step1 = {'I': 0, 'II/IIF/III': 1, 'IV': 2}
class_dict_step2 = {'II': 0, 'IIF': 1, 'III': 2}

def TestData(args):
    data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    image_dataset = SegSliceDataSet(rf'F:\renal_cyst\testdata\{args.testdata}',
                                    rf'F:\renal_cyst\gt\{args.testdata}', 'test',
                                    data_transforms)
    dataloders = torch.utils.data.DataLoader(image_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers,
                                             shuffle=True)
    dataset_sizes = len(image_dataset)
    return dataloders, dataset_sizes


def KfoldData(args, fold):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomCrop(size=224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            # Gaussian_noise(mean=0.5, sigma=0.1),
            # AddGaussianNoise(),
            # transforms.RandomAffine(degrees=30),
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]),
    }
    data_dir = r'F:\renal_cyst\dataset'
    mask_dir = r'F:\renal_cyst\gt\native'
    file_list, labels = Kfold(data_dir)
    print(labels)
    labels.pop(fold)
    val_list = file_list.pop(fold)
    train_list = functools.reduce(operator.concat, file_list)
    num = {'I': 0, 'II': 0, 'IIF': 0, 'III': 0, 'IV': 0}
    for i in labels:
        num['I'] += i['I']
        num['II'] += i['II']
        num['IIF'] += i['IIF']
        num['III'] += i['III']
        num['IV'] += i['IV']
    num['I'] = num['IV']
    w2s = num['II'] + num['IIF'] + num['III']
    w1s = num['I'] + w2s + num['IV']
    w1 = [w1s / num['I'], w1s / w2s, w1s / num['IV']]
    w2 = [w2s / num['II'], w2s / num['IIF'], w2s / num['III']]

    image_datasets = {}
    image_datasets['train'] = list2dataset(data_dir, mask_dir, train_list, data_transforms['train'], 'train')
    image_datasets['val'] = list2dataset(data_dir, mask_dir, val_list, data_transforms['val'], 'val')
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=args.batch_size,
                                                 num_workers=args.num_workers,
                                                 shuffle = True) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    return dataloders, dataset_sizes, w1, w2


def randomData(args):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]),
    }

    data_dir = r'F:\renal_cyst\dataset'
    mask_dir = r'F:\renal_cyst\gt\native'
    files, labels = getRandom(data_dir)
    print(labels)
    num = {}
    num['I'] = labels['I']
    num['II'] = labels['II']
    num['IIF'] = labels['IIF']
    num['III'] = labels['III']
    num['IV'] = labels['IV']
    num['I'] = num['IV']
    w2s = num['II'] + num['IIF'] + num['III']
    w1s = num['I'] + w2s + num['IV']
    w1 = [w1s / num['I'], w1s / w2s, w1s / num['IV']]
    w2 = [w2s / num['II'], w2s / num['IIF'], w2s / num['III']]

    image_datasets = {}
    image_datasets['train'] = list2dataset(data_dir, mask_dir, files['train'], data_transforms['train'], 'train')
    image_datasets['val'] = list2dataset(data_dir, mask_dir, files['val'], data_transforms['val'], 'val')
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=args.batch_size,
                                                 num_workers=args.num_workers,
                                                 shuffle=True) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    return dataloders, dataset_sizes, w1, w2


def SliceData(args):
# data_transform, pay attention that the input of Normalize() is Tensor and the input of RandomResizedCrop() or RandomHorizontalFlip() is PIL Image
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomCrop(size=224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            # Gaussian_noise(mean=0.5, sigma=0.1),
            # AddGaussianNoise(),
            # transforms.RandomAffine(degrees=30),
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]),
    }
    image_datasets = {}
    image_datasets['train'] = SegSliceDataSet(r'F:\renal_cyst\dataset',
                                              r'F:\renal_cyst\gt\native', 'train',
                                              data_transforms['train'])
    image_datasets['val'] = SegSliceDataSet(r'F:\renal_cyst\dataset',
                                            r'F:\renal_cyst\gt\native', 'val',
                                            data_transforms['val'])

    # image_datasets['train'] = DetectedSliceDataSet(r'F:\renal_cyst\trainingData\detection\coco\train2017', 'train',data_transforms['train'])
    # image_datasets['val'] = DetectedSliceDataSet(r'F:\renal_cyst\trainingData\detection\coco\val2017', 'test', data_transforms['val'])

    # WeightSampler
    # counts = [287, 84, 51, 64, 124]
    # weight = [sum(counts) / c for c in counts]
    # sampler = WeightedRandomSampler(weight, args.batch_size)
    # dataloders = {'train': torch.utils.data.DataLoader(image_datasets['train'],
    #                                              batch_size=args.batch_size,
    #                                              num_workers=args.num_workers,
    #                                              sampler=sampler,
    #                                              shuffle=False),
    #               'val': torch.utils.data.DataLoader(image_datasets['val'],
    #                                              batch_size=args.batch_size,
    #                                              num_workers=args.num_workers,
    #                                              shuffle=False)}
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=args.batch_size,
                                                 num_workers=args.num_workers,
                                                 shuffle = True) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    return dataloders, dataset_sizes


class Gaussian_noise(object):
    def __init__(self, mean, sigma, p=0.5):
        self.mean = mean
        self.sigma = sigma
        self.p = p

    def __call__(self, img):
        rand = random.randint(0, 1)
        if rand > self.p:
            img_ = np.array(img).copy()
            noise = np.random.normal(self.mean, self.sigma, img_.shape)
            gaussian_out = img_ + noise
            return Image.fromarray(gaussian_out)
        else:
            return img


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, img):
        tensor = transforms.ToTensor()(img)
        tensor = tensor + torch.randn(tensor.size()) * self.std + self.mean
        return transforms.ToPILImage()(tensor)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def read_file(file):  # 读取niifile文件
    img = sitk.GetArrayFromImage(sitk.ReadImage(file))[0,:,:]
    return img


def EvenInt(num):
    if (int(num) % 2) == 0:
        return int(num)
    else:
        return int(num) + 1


def Resize(width,height):
    if width>height:
        return (224,EvenInt(224*height/width))
    else:
        return (EvenInt(224*width/height),224)


def normalize(in_array):
    img_max = in_array.max()
    img_min = in_array.min()
    img_range = img_max - img_min
    out_array = (in_array - img_min) / (img_range + 1e-6)
    return out_array


def windows(img_arr, wl, ww):
    wmax = wl + 0.5 * ww
    wmin = wl - 0.5 * ww
    img_arr[img_arr > wmax] = wmax
    img_arr[img_arr < wmin] = wmin
    return img_arr


def roi_expend(roi, alpha=1):
    x0, y0, x1, y1 = roi
    w = x1 - x0
    h = y1 - y0
    new_x0 = int(x0 - (alpha - 1) * w / 2)
    new_x1 = int(x1 + (alpha - 1) * w / 2)
    new_y0 = int(y0 - (alpha - 1) * h / 2)
    new_y1 = int(y1 + (alpha - 1) * h / 2)
    return [new_x0, new_y0, new_x1, new_y1]


def circle(image):
    h, w = image.shape
    img_zero = np.zeros((h, w))
    cv2.ellipse(img_zero, (int(w/2), int(h/2)), (int(w/2), int(h/2)), 0, 0, 360, 1, -1)
    return image * img_zero


def Resize_Padding(in_array):
    resized_shape = Resize(in_array.shape[0], in_array.shape[1])
    PIL_image = Image.fromarray(in_array)
    resized_array = torchvision.transforms.Resize(resized_shape)(PIL_image)
    (rows, cols) = resized_shape
    padding_array = np.zeros([224, 224])
    Top_padding = int((224 - resized_shape[0])/2)
    Left_padding = int((224 - resized_shape[1])/2)
    padding_array[Top_padding:rows + Top_padding, Left_padding:cols + Left_padding] = resized_array
    out_image = Image.fromarray(padding_array)
    return out_image


def concat_channels(image, mask):
    img = image * mask
    img[img < -20] = -20
    img[img > 500] = 500
    img = (img + 20) / 520
    image = normalize(image)
    ch1 = Resize_Padding(image * 255).convert('L')
    ch2 = Resize_Padding(img * 255).convert('L')
    ch3 = Resize_Padding(img * 255).convert('L')
    out_image = Image.merge('RGB', [ch1, ch2, ch3])
    # image = torch.from_numpy(image).unsqueeze(0)
    # img = torch.from_numpy(img).unsqueeze(0)
    # out_image = torch.cat((image, img), 0)
    return out_image


def add_noise(img, mean, std):
    noise = np.random.normal(mean, std, size=img.shape)
    return img + noise


# class DetectedSliceDataSet(torch.utils.data.Dataset):
#     def __init__(self, root_dir, mode, data_transforms):
#         self.img_path = os.listdir(root_dir)
#         self.data_transforms = data_transforms
#         self.mode = mode
#         self.root_dir = root_dir
#         self.imgs = self._make_dataset()
#
#     def __len__(self):
#         return len(self.imgs)
#
#     def __getitem__(self, item):
#         data, label, roi, mask = self.imgs[item]
#         img = np.load(data)
#         mask_array = sitk.GetArrayFromImage(sitk.ReadImage(mask)).squeeze()
#         img = img * mask_array
#         # img = windows(img, wl=30, ww=300)
#         # roi = roi_expend(roi, alpha=1.2)
#         x_0, y_0, x_1, y_1 = roi
#         cropped_region = img[y_0:y_1, x_0:x_1]
#         # img_show = Image.fromarray(normalize(cropped_region) * 255)
#         # img_show.show()
#         cropped_region[cropped_region < -20] = -20
#         cropped_region[cropped_region > 500] = 500
#         cropped_region = (cropped_region + 20) / 520
#         # cropped_region_show = Image.fromarray(cropped_region * 255)
#         # cropped_region_show.show()
#         # circled_region = circle(cropped_region)
#         # circled_region_show = Image.fromarray(circled_region * 255)
#         # circled_region_show.show()
#         PIL_image = Resize_Padding(cropped_region)
#         out = self.data_transforms(PIL_image)
#         return out, label
#
#     def _make_dataset(self):
#         images = []
#         mask_path = r'C:\Users\admin\Desktop\NF5.5'
#         df = pd.read_csv(os.path.join('F:\\renal_cyst\\dataset', self.mode + '_labels.csv'), header=None, names=['file', 'x_0', 'y_0', 'x_1', 'y_1', 'label'])
#         if not os.path.exists(self.root_dir):
#             return -1
#         for fname in os.listdir(self.root_dir):
#             if fname.endswith('.npy'):
#                 info_list = os.path.splitext(fname)[0].split('_')
#                 # if info_list[1] != 'IIF' and info_list[1] != 'III':
#                 #     continue
#                 focus_class = class_dict[info_list[1]]
#                 mask = os.path.join(os.path.join(mask_path, info_list[0]),
#                                     info_list[0] + '_' + info_list[1] + '_' + re.findall(r'\d+', fname.split('_')[-1])[0] + '.nii')
#                 file_path = os.path.join(self.root_dir, fname)
#                 df_file = df.loc[df['file'] == fname]
#                 roi = [df_file['x_0'].values[0], df_file['y_0'].values[0], df_file['x_1'].values[0], df_file['y_1'].values[0]]
#                 item = (file_path, focus_class, roi, mask)
#                 images.append(item)
#         return images


class SegSliceDataSet(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, mode, data_transforms):
        self.mask_dir = mask_dir
        self.image_dir = image_dir
        self.mode = mode
        self.data_transforms = data_transforms
        self.imgs = self._make_dataset()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        img_path, label, ex_label, mask_path, fname = self.imgs[item]
        image = np.load(img_path)
        if mask_path.endswith('.nii.gz'):
            mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).squeeze()
        elif mask_path.endswith('.npy'):
            mask = np.load(mask_path)
        else:
            raise Exception(f'wrong mask path: {mask_path}')
        # image = zoom(image, (512 / image.shape[0], 512 / image.shape[1]))
        # mask = zoom(mask, (512 / mask.shape[0], 512 / mask.shape[1]))
        # image = add_noise(image, mean=0.5, std=0.1)
        # image[image < -20] = -20
        # image[image > 500] = 500
        # image = (image + 20) / 520
        # plt.imshow(image, cmap='gray')
        # plt.show()
        # plt.imshow(mask, cmap='gray')
        # plt.show()
        img = image * mask
        # plt.imshow(img, cmap='gray')
        # plt.show()
        # min_ = img.min()
        # img[img == 0] = min_
        # img = normalize(img)
        # img = resize(img, output_shape=(256, 256))
        # x, y = np.where(img != 0)
        # img = img[x.min():x.max(), y.min():y.max()]
        img[img < -20] = -20
        img[img > 500] = 500
        img = (img + 20) / 520
        # info = fname.split('_')
        # if info[2] == 'MUMC':
        #     n = 3
        #     # window = np.ones((n, n)) / n ** 2
        #     img = median_filter(img, (n, n))
        # save_path = './results/III_label/'
        # os.makedirs(save_path, exist_ok=True)
        # plt.figure()
        # plt.imshow(img, cmap='gray')
        # save_name = save_path + fname + '_' + f'{label}' + '.png'
        # plt.savefig(save_name)
        # plt.close()
        # PIL_image = Resize_Padding(img)
        # PIL_image = Image.fromarray(img)
        # PIL_image = concat_channels(image, mask)
        # array = np.array(PIL_image)
        # plt.imshow(array, cmap='gray')
        # plt.title('before')
        # plt.show()
        # array = np.expand_dims(array, axis=0)
        # array = np.expand_dims(array, axis=0)
        # out = augmentation_func(array)
        # out = out[0, :, :, :]
        PIL_image = Image.fromarray(img)
        out = self.data_transforms(PIL_image)
        # plt.subplot(121)
        # plt.imshow(img, cmap='gray')
        # plt.title('before')
        # plt.subplot(122)
        # plt.imshow(out.numpy()[0, :, :], cmap='gray')
        # # plt.imshow(np.concatenate((img, out.numpy()[0, :, :]), axis=1), cmap='gray')
        # plt.title('after')
        # plt.show()
        return out, label, ex_label, fname

    def _make_dataset(self):
        images = []
        I_num = 0
        df = pd.read_csv(os.path.join(self.image_dir, self.mode + '_labels.csv'),
                         header=None, names=['file', 'x_0', 'y_0', 'x_1', 'y_1', 'label'])
        df = df.drop_duplicates(subset=['file'])
        multilabel_df = pd.read_csv(r'F:\renal_cyst\multilabel.csv')
        # if not os.path.exists(self.mask_dir):
        #     return -1
        labels_num = np.zeros(len(class_dict))
        for fname in df['file']:
            info_list = fname.split('.')[0].split('_')
            # if info_list[1] in ['II', 'IIF', 'III']:
            #     info_list[1] = 'II/IIF/III'
            # if info_list[1] not in ['II', 'IIF', 'III']:
            #     continue
            # if info_list[1] not in ['III']:
            #     continue
            # if info_list[2] != 'MUMC':
            #     continue

            # if info_list[1] == 'I':
            #     if self.mode == 'train':
            #         if I_num >= 290:
            #             continue
            #         else:
            #             I_num += 1
            #     else:
            #         if I_num >= 45:
            #             continue
            #         else:
            #             I_num += 1

            focus_class = class_dict[info_list[1]]
            labels_num[focus_class] += 1
            ex_label = [-1, -1, -1, -1, -1]
            # if info_list[1] in ['I', 'IV']:
            #     ex_label = [-1, -1, -1, -1, -1]
            # else:
            #     multi_df = multilabel_df[multilabel_df['case'] == int(info_list[0])]
            #     multi_df = multi_df[multi_df['slice'] == info_list[2]]
            #     if len(multi_df) == 0:
            #         continue
            #     septa = multi_df['septa'].item()
            #     septa_tn = multi_df['septa thickness'].item()
            #     wall_tn = multi_df['wall thickness'].item()
            #     nodule = multi_df['nodule'].item()
            #     calcification = multi_df['calcification'].item()
            #     ex_label = [septa, septa_tn, wall_tn, nodule, calcification]
            image_path = os.path.join(self.image_dir, fname)
            if info_list[-1] == 'copy':
                fname = fname.replace('_copy', '')
            # mask = np.load(gt_path)
            # image = np.load(image_path)
            # img = image * mask
            # if img.max() > 70:
            #     continue
            # print(img.max())
            mask_path = os.path.join(self.mask_dir, fname)
            item = (image_path, focus_class, ex_label, mask_path, fname.split('.')[0])
            # mask_path = os.path.join(self.mask_dir, fname.replace('.npy', '.nii.gz'))
            # mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).squeeze()
            # if mask.sum() == 0:
            #     continue
            # item = (image_path, focus_class, ex_label, mask_path, fname.split('.')[0])
            images.append(item)
        print(labels_num)
        return images


class list2dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, img_list, data_transforms, mode):
        self.img_list = img_list
        self.mask_dir = mask_dir
        self.image_dir = image_dir
        self.data_transforms = data_transforms
        self.mode = mode
        self.imgs = self._make_dataset()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        img_path, label, ex_label, mask_path, fname = self.imgs[item]
        image = np.load(img_path)
        if mask_path.endswith('.nii.gz'):
            mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).squeeze()
        elif mask_path.endswith('.npy'):
            mask = np.load(mask_path)
        else:
            raise Exception(f'wrong mask path: {mask_path}')
        img = image * mask
        img[img < -20] = -20
        img[img > 500] = 500
        img = (img + 20) / 520
        PIL_image = Image.fromarray(img)
        out = self.data_transforms(PIL_image)
        return out, label, ex_label, fname

    def _make_dataset(self):
        images = []
        I_num = 0
        multilabel_df1 = pd.read_csv(r'F:\renal_cyst\multilabel.csv')
        multilabel_df2 = pd.read_csv(r'F:\NF 7.20\补充数据辅助标签.csv')
        multilabel_df = pd.concat([multilabel_df1, multilabel_df2])
        labels_num = np.zeros(len(class_dict))
        for fname in self.img_list:
            info_list = fname.split('.')[0].split('_')
            # if info_list[1] in ['II', 'IIF', 'III']:
            #     info_list[1] = 'II/IIF/III'
            # if info_list[1] not in ['II', 'IIF', 'III']:
            #     continue
            # if info_list[1] == 'I':
            #     if self.mode == 'train':
            #         if I_num >= 360:
            #             continue
            #         else:
            #             I_num += 1
            #     else:
            #         if I_num >= 103:
            #             continue
            #         else:
            #             I_num += 1
            focus_class = class_dict[info_list[1]]
            labels_num[focus_class] += 1
            ex_label = [-1, -1, -1, -1, -1]
            # if info_list[1] in ['I', 'IV']:
            #     ex_label = [-1, -1, -1, -1, -1]
            # else:
            #     multi_df = multilabel_df[multilabel_df['case'] == int(info_list[0])]
            #     multi_df = multi_df[multi_df['slice'] == int(info_list[2])]
            #     if len(multi_df) == 0:
            #         continue
            #     septa = multi_df['septa'].item()
            #     septa_tn = multi_df['septa thickness'].item()
            #     wall_tn = multi_df['wall thickness'].item()
            #     nodule = multi_df['nodule'].item()
            #     calcification = multi_df['calcification'].item()
            #     ex_label = [septa, septa_tn, wall_tn, nodule, calcification]
            image_path = os.path.join(self.image_dir, fname)
            if info_list[-1] == 'copy':
                fname = fname.replace('_copy', '')
            mask_path = os.path.join(self.mask_dir, fname)
            item = (image_path, focus_class, ex_label, mask_path, fname.split('.')[0])
            images.append(item)
        print(labels_num)
        return images


class SegSliceDataSet_step1(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, mode, data_transforms):
        self.mask_dir = mask_dir
        self.image_dir = image_dir
        self.mode = mode
        self.data_transforms = data_transforms
        self.imgs = self._make_dataset()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        img_path, label, mask_path, fname = self.imgs[item]
        image = np.load(img_path)
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).squeeze()
        img = image * mask
        x, y = np.where(img != 0)
        img = img[x.min():x.max(), y.min():y.max()]
        img[img < -20] = -20
        img[img > 500] = 500
        img = (img + 20) / 520
        # img = normalize(img)
        PIL_image = Resize_Padding(img)
        out = self.data_transforms(PIL_image)
        return out, label, fname

    def _make_dataset(self):
        images = []
        df = pd.read_csv(os.path.join(self.image_dir, self.mode + '_labels.csv'),
                         header=None, names=['file', 'x_0', 'y_0', 'x_1', 'y_1', 'label'])
        if not os.path.exists(self.mask_dir):
            return -1
        labels_num = np.zeros(len(class_dict))
        for fname in df['file']:
            if fname.endswith('.npy'):
                info_list = os.path.splitext(fname)[0].split('_')
                if info_list[1] in ['II', 'IIF', 'III']:
                    info_list[1] = 'II/IIF/III'
                # if info_list[1] in ['I', 'IV']:
                #     continue
                focus_class = class_dict_step1[info_list[1]]
                labels_num[focus_class] += 1
                image_path = os.path.join(self.image_dir, fname)
                if info_list[-1] == 'copy':
                    fname = fname.replace('_copy', '')
                mask_path = os.path.join(self.mask_dir, fname.replace('.npy', '.nii.gz'))
                mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).squeeze()
                if mask.sum() == 0:
                    continue
                item = (image_path, focus_class, mask_path, fname.split('.')[0])
                images.append(item)
        print(labels_num)
        return images


def PatientData(args):
# data_transform, pay attention that the input of Normalize() is Tensor and the input of RandomResizedCrop() or RandomHorizontalFlip() is PIL Image
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30, resample=False, expand=False, center=None),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
        ]),
    }
    dataset_sizes = {}
    dataloders = {}
    patient_list = []
    dir_path = 'F:\\renal_cyst\\trainingData\\detection\\coco\\test'
    for patient in os.listdir(dir_path):
        patient_dir = os.path.join(dir_path, patient)
        image_dataset = DetectedSliceDataSet(patient_dir, 'test', data_transforms['val'])
        # wrap your data and label into Tensor
        dataloder = torch.utils.data.DataLoader(image_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers)
        dataset_size = len(image_dataset)

        patient_list.append(patient)
        dataset_sizes[patient] = dataset_size
        dataloders[patient] = dataloder

    return patient_list, dataloders, dataset_sizes