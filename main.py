import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from argparse import ArgumentParser
import os
from datetime import datetime

from model import *
from datasets import *
from increment import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(labeled_dataset, val_dataset, unlabeled_dataset):

    train_loader = DataLoader(labeled_dataset, batch_size=args.bs, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)

    model = Net().to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    now = datetime.utcnow()
    with open('results/{}_logs_{}.csv'.format(args.mode, now), 'w') as f:
        f.write('epoch, number_of_data, train_loss, train_accuracy, validation_loss, validation_accuracy\n')
        f.close()

    for e in range(args.epoch):
        model.train()
        total = 0
        correct = 0
        train_loss_sum = 0.0
        train_accuracy = 0

        # if not e == 0:
        #     labeled_dataset, unlabeled_dataset = add_data(labeled_dataset, unlabeled_dataset, args.mode, args.add_num)

        train_loader = DataLoader(labeled_dataset, batch_size=args.bs, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)

        for i, (input, target) in enumerate(train_loader):
            input = input.to(device)
            target = target.to(device)

            y_pred = model(input)

            total += len(target)

            correct += (y_pred.argmax(1) == target).sum().item()

            train_loss = criterion(y_pred, target)
            train_loss_sum += train_loss

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        train_accuracy = correct/total
        print('----------------------------------------------------')
        print('Training >> Epoch : [{}/{}] Loss : {} Accuracy {}'.format(e, args.epoch, train_loss_sum, train_accuracy))

        val_loss_sum, val_accuracy = validation(model, val_loader, criterion)
        print('Test >> Epoch : [{}/{}] Loss : {} Accuracy {}'.format(e, args.epoch, val_loss_sum, val_accuracy))

        with open('results/{}_logs_{}.csv'.format(args.mode, now), 'a') as f:
            f.write('{},{},{},{},{},{}\n'.format(e, len(labeled_dataset), train_loss_sum, train_accuracy, val_loss_sum, val_accuracy))
            f.close()


def validation(model, val_loader, criterion):
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        val_loss_sum = 0.0

        for i, (input, target) in enumerate(val_loader):
            input = input.to(device)
            target = target.to(device)

            y_pred = model(input)

            total += len(target)
            correct += (y_pred.argmax(1) == target).sum().item()
            val_loss_sum += criterion(y_pred, target)

        accuracy = correct/total

        return val_loss_sum, accuracy


def main(args):

    if args.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        org_dataset = datasets.MNIST('./data', train=True, download=True)   #we will transform it when making using dataset
        val_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    ##############TODO#############3
    '''
    dataset without transform,
    transform in Split Dataset --> it can be easier when.... you utilize increment and augmentation for AL
    
    Q1. How to AL? --> do I need to add unlabeled dataset to labeled_dataset? How and When?    
    '''
    ###############END##############3

    labeled_dataset, unlabeled_dataset = init_dataset(org_dataset, args.init_num)

    train(labeled_dataset, val_dataset, unlabeled_dataset)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--mode', choices=['random', 'augment','choice'], default='random')
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--learning_rate', dest='lr', default=1e-4)
    parser.add_argument('--epoch', default=50)
    parser.add_argument('--add_num', default=100)
    parser.add_argument('--init_num', default=100)
    parser.add_argument('--batch_size', dest='bs', default=64)

    args = parser.parse_args()

    if not os.path.exists('results/'):
        os.makedirs('results/')

    main(args)