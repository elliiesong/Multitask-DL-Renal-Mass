import torch
import torchio as tio
from torch.nn import Module
import matplotlib.pyplot as plt

def augmentation_func(img):
    tioimg = tio.ScalarImage(tensor=img)
    subject = tio.Subject(images=tioimg)
    # img = torch.Tensor(img)
    # img = ToTensor3D(normalize=False)(img)

    # tioimg = tio.ScalarImage(tensor=img)
    transforms = []
    # # spatial augmentation
    spatial_aug = {}
    spatial_aug[tio.RandomFlip()] = 0.3
    # spatial_aug[tio.RandomAffine()] = 0.3
    # # time cost over 2s
    # spatial_aug[tio.RandomElasticDeformation()] = 0.1
    # # time consume a lot
    # spatial_aug[tio.RandomAnisotropy()] = 0.1
    # # # time costs 0.06s for 64, 512, 512, 100 iterations
    # spatial_aug[tio.RandomSwap()] = 0.1

    # intensity augmentation
    intensity_aug = {}
    # # time costs 2s for 64, 512, 512
    intensity_aug[tio.RandomMotion()] = 0.1
    # time costs 1s for 64, 512, 512
    intensity_aug[tio.RandomGhosting()] = 0.01
    # # time costs 1s for 64, 512, 512, for MRI
    intensity_aug[tio.RandomSpike()] = 0.01
    # # time costs 4s for 64, 512, 512, for MRI
    # intensity_aug[tio.RandomBiasField()] = 0.05
    # # # time costs 0.3s for 64, 512, 512
    intensity_aug[tio.RandomBlur()] = 0.2
    # # # time costs 0.2s for 64, 512, 512
    intensity_aug[tio.RandomNoise()] = 0.3
    # # # time costs 0.08s for 64, 512, 512
    intensity_aug[tio.RandomGamma()] = 0.1
    #
    spatial_aug = tio.OneOf(spatial_aug)
    transforms.append(spatial_aug)

    intensity_aug = tio.OneOf(intensity_aug)
    transforms.append(intensity_aug)

    # transforms.append(tio.EnsureShapeMultiple(2 ** 4))
    #
    # # normalization
    # transforms.append(tio.RescaleIntensity(out_min_max=(0, 1)))
    # transforms.append(tio.ZNormalization())
    # print('transforms', transforms)
    transforms = tio.Compose(transforms)
    # print('transforms', transforms)
    # print('img', img.shape)
    subject = transforms(subject)
    # history = subject.get_composed_history()
    # print('history', history)
    image = subject['images'][tio.DATA]
    # print('image', image.shape)
    # plt.imshow(image[0, 40, :, :], cmap='gray')
    # plt.show()
    # print('image', image.shape)
    return image

