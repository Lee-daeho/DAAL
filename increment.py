from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torch

import numpy as np
import cv2

from datasets import SplitDataset
from augment import *

def add_data(labeled_dataset, unlabeled_dataset,  mode, add_num, chosen_dataset= None):

    if mode == 'random':
        random_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        permutes = np.random.permutation(len(unlabeled_dataset))

        L_index = permutes[:add_num]
        U_index = permutes[add_num:]

        labled_data = torch.cat((labeled_dataset.data, unlabeled_dataset.data[L_index]), 0)
        labled_label = torch.cat((labeled_dataset.label, unlabeled_dataset.label[L_index]), 0)

        unlabeled_data = unlabeled_dataset.data[U_index]
        unlabeled_label = unlabeled_dataset.label[U_index]

        labeled_dataset = SplitDataset(labled_data, labled_label, random_transform)
        unlabeled_dataset = SplitDataset(unlabeled_data, unlabeled_label, random_transform)

        return labeled_dataset, unlabeled_dataset

    if mode == 'augment':

        base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])


        added_data = torch.Tensor([])
        added_label = np.array([])
        chosen_data = torch.Tensor([])
        chosen_label = np.array([])

        for idx in range(len(labeled_dataset)):
            data = labeled_dataset[idx][0]
            label = labeled_dataset.label[idx]

            if len(data.size()) == 2:
                data = data.unsqueeze(0)

            added_data = torch.cat((added_data, data), dim=0)
            added_label = np.append(added_label, label.numpy())
            chosen_data = torch.cat((chosen_data, data), dim=0)
            chosen_label = np.append(chosen_label, label.numpy())
            if idx >= len(labeled_dataset)-add_num: #adapt transform to only added data
                transform_list = augment(labeled_dataset[0][0].size(1))
                for transform in transform_list:

                    augmented = transform(data)

                    tr = UnNormalize((0.1307,), (0.3081,))
                    img_chk = np.array(tr(augmented.permute(1, 2, 0)) * 255, dtype=np.uint8)

                    while True:
                        cv2.imshow('{}'.format(label), img_chk)

                        if cv2.waitKey(1) == ord('o'):
                            chosen_data = torch.cat((chosen_data, augmented), dim=0)
                            chosen_label = np.append(chosen_label, label.numpy())
                            cv2.destroyAllWindows()
                            break
                        elif cv2.waitKey(1) == ord('n'):
                            cv2.destroyAllWindows()
                            break

                    added_data = torch.cat((added_data, augmented), dim=0)
                    added_label = np.append(added_label, label.numpy())

        permutes = np.random.permutation(len(unlabeled_dataset))

        L_indexes = permutes[:add_num]
        U_indexes = permutes[add_num:]

        for idx in L_indexes:
            added_data = torch.cat((added_data, base_transform(unlabeled_dataset.data[idx])), dim=0)
            chosen_data = torch.cat((chosen_data, base_transform(unlabeled_dataset.data[idx])), dim=0)

        added_label = np.append(added_label, unlabeled_dataset.label[L_indexes].numpy())
        chosen_label = np.append(chosen_label, unlabeled_dataset.label[L_indexes].numpy())

        labeled_dataset = SplitDataset(added_data, torch.Tensor(added_label).long())
        chosen_dataset = SplitDataset(chosen_data, torch.Tensor(chosen_label).long())

        unlabeled_dataset = SplitDataset(unlabeled_dataset.data[U_indexes], unlabeled_dataset.label[U_indexes], base_transform)

        print(labeled_dataset.data.shape)
        print(chosen_dataset.data.shape)

        return labeled_dataset, unlabeled_dataset, chosen_dataset



