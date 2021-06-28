from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torch

import numpy as np

from datasets import SplitDataset
from augment import *

def add_data(labeled_dataset, unlabeled_dataset, mode, add_num):

    if mode == 'random':
        permutes = np.random.permutation(len(unlabeled_dataset))

        L_index = permutes[:add_num]
        U_index = permutes[add_num:]

        labled_data = torch.cat((labeled_dataset.data, unlabeled_dataset.data[L_index]), 0)
        labled_label = torch.cat((labeled_dataset.label, unlabeled_dataset.label[L_index]), 0)

        unlabeled_data = unlabeled_dataset.data[U_index]
        unlabeled_label = unlabeled_dataset.label[U_index]

        labeled_dataset = SplitDataset(labled_data, labled_label)
        unlabeled_dataset = SplitDataset(unlabeled_data, unlabeled_label)

        return labeled_dataset, unlabeled_dataset

    if mode == 'augment':
        transform_list = augment(labeled_dataset.shape(2))

        for datas in labeled_dataset:
            for transform in transform_list:
                augmented = transform(datas.data)