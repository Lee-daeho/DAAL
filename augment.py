from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torch

import numpy as np

from utils import *

def augment(size):

    cc_size = np.random.randint(1,size)

    verticalflip = transforms.Compose([
        UnNormalize((0.1307,), (0.3081,)),
        transforms.ToPILImage(),
        transforms.RandomVerticalFlip(p=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    horizontalflip = transforms.Compose([
        UnNormalize((0.1307,), (0.3081,)),
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    centercrop = transforms.Compose([
        UnNormalize((0.1307,), (0.3081,)),
        transforms.ToPILImage(),
        transforms.CenterCrop(cc_size),
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    randomaffine = transforms.Compose([
        UnNormalize((0.1307,), (0.3081,)),
        transforms.ToPILImage(),
        transforms.RandomAffine(180, translate=(0.5,0.5),shear=180),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    randomcrop = transforms.Compose([
        UnNormalize((0.1307,), (0.3081,)),
        transforms.ToPILImage(),
        transforms.RandomCrop(size, pad_if_needed=True),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    randomrotation = transforms.Compose([
        UnNormalize((0.1307,), (0.3081,)),
        transforms.ToPILImage(),
        transforms.RandomRotation(180, fill=(0,)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    randomerase = transforms.Compose([
        transforms.RandomErasing(p=1)
    ])  #done on Tensor


    return [verticalflip, horizontalflip, centercrop]#, randomaffine, randomcrop, randomrotation, randomerase]