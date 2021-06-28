from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torch

import numpy as np

def augment(size):

    cc_size = np.random.randint(0,size)

    verticalflip = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomVerticalFlip(p=1),
        transforms.ToTensor()
    ])

    horizontalflip = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=1),
        transforms.ToTensor()
    ])

    centercrop = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(cc_size),
        transforms.ToTensor()
    ])

    randomaffine = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine(180, translate=(0.5,0.5),shear=180),
        transforms.ToTensor()
    ])

    randomcrop = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(size, pad_if_needed=True),
        transforms.ToTensor()
    ])

    randomrotation = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(180, fill=(0,)),
        transforms.ToTensor()
    ])

    randomerase = transforms.Compose([
        transforms.RandomErasing(p=1)
    ])  #done on Tensor

    # randomadd = transforms.Compose([
    #     transforms.ToPILImage(),
    #
    # ])

    return [verticalflip, horizontalflip, centercrop, randomaffine, randomcrop, randomrotation, randomerase]