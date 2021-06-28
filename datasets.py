from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torch

import numpy as np

org_dataset = datasets.MNIST('./data', train=True, download=True)
test_dataset =  datasets.MNIST('./data', train=False, download=True)


class SplitDataset(Dataset):
    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):

        if self.transform:
            return self.transform(self.data[idx]), self.label[idx]

        return self.data[idx], self.label[idx]
        # self.transform({'input': torch.tensor(self.data[idx], dtype=torch.float32),
        #         'label': torch.tensor(self.label[idx], dtype=torch.long)})


def init_dataset(full_data, init_num):

    permutes = np.random.permutation(len(full_data))

    L_indexes = permutes[:init_num]
    U_indexes = permutes[init_num:]

    labeled_dataset = SplitDataset(full_data.data[L_indexes], full_data.targets[L_indexes])
    unlabeled_dataset = SplitDataset(full_data.data[U_indexes], full_data.targets[U_indexes])

    return labeled_dataset, unlabeled_dataset
