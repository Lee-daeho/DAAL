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
from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(labeled_dataset, val_dataset, unlabeled_dataset):

    chosen_dataset = None
    unchosen_dataset = None

    model = Net().to(device)
    model_chosen = Net().to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    criterion_chosen = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer_chosen = torch.optim.Adam(model_chosen.parameters(), lr=args.lr)
    now = datetime.utcnow()

    with open('results/{}_normal_logs_{}.csv'.format(now, args.mode), 'w') as f:
        f.write('epoch, number_of_data, train_loss, train_accuracy, validation_loss, validation_accuracy\n')
        f.close()
    with open('results/{}_chosen_logs_{}.csv'.format(now, args.mode), 'w') as f:
        f.write('epoch, number_of_data, train_loss, train_accuracy, validation_loss, validation_accuracy\n')
        f.close()

    for e in range(args.epoch):
        model.train()
        total = 0
        correct = 0
        train_loss_sum = 0.0
        train_accuracy = 0

        if not e == 0:
            if args.mode == 'random':
                labeled_dataset, unlabeled_dataset = add_data(labeled_dataset, unlabeled_dataset,
                                                              args.mode, args.add_num)
            if args.mode == 'augment':
                if not chosen_dataset:
                    labeled_dataset, unlabeled_dataset, chosen_dataset = \
                        add_data(labeled_dataset, unlabeled_dataset, args.mode, args.add_num)
                else:
                    labeled_dataset, unlabeled_dataset, chosen_dataset = \
                        add_data(labeled_dataset, unlabeled_dataset, args.mode, args.add_num, chosen_dataset)

        train_loader = DataLoader(labeled_dataset, batch_size=args.bs, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)

        if chosen_dataset:
            chosen_loader = DataLoader(chosen_dataset, batch_size=args.bs, shuffle=False)
        else:
            chosen_loader = DataLoader(labeled_dataset, batch_size=args.bs, shuffle=False)

        for i, (input, target) in enumerate(train_loader):
            input = input.to(device)
            target = target.to(device)

            if len(input.size()) == 3:
                input = input.unsqueeze(1)
            elif len(input.size()) == 4:
                pass
            else:
                return 'something wrong with input'

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

        with open('results/{}_normal_logs_{}.csv'.format(now, args.mode), 'a') as f:
            f.write('{},{},{},{},{},{}\n'.format(e, len(labeled_dataset), train_loss_sum, train_accuracy, val_loss_sum, val_accuracy))
            f.close()

    ###########################chosen train#################################
        if args.mode == 'augment':
            for i, (input, target) in enumerate(chosen_loader):
                input = input.to(device)
                target = target.to(device)

                if len(input.size()) == 3:
                    input = input.unsqueeze(1)
                elif len(input.size()) == 4:
                    pass
                else:
                    return 'something wrong with input'

                y_pred = model_chosen(input)

                total += len(target)

                correct += (y_pred.argmax(1) == target).sum().item()

                train_loss = criterion_chosen(y_pred, target)
                train_loss_sum += train_loss

                optimizer_chosen.zero_grad()
                train_loss.backward()
                optimizer_chosen.step()

            train_accuracy = correct/total
            print('----------------------------------------------------')
            print('Chosen Training >> Epoch : [{}/{}] Loss : {} Accuracy {}'.format(e, args.epoch, train_loss_sum, train_accuracy))

            val_loss_sum, val_accuracy = validation(model_chosen, val_loader, criterion_chosen)
            print('Chosen Test >> Epoch : [{}/{}] Loss : {} Accuracy {}'.format(e, args.epoch, val_loss_sum, val_accuracy))
            print('----------------------------------------------------')

            with open('results/{}_chosen_logs_{}.csv'.format(now, args.mode), 'a') as f:
                f.write('{},{},{},{},{},{}\n'.format(e, len(chosen_dataset) if chosen_dataset else len(labeled_dataset),
                                                     train_loss_sum, train_accuracy, val_loss_sum, val_accuracy))
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

    labeled_dataset, unlabeled_dataset = init_dataset(org_dataset, args.init_num)

    train(labeled_dataset, val_dataset, unlabeled_dataset)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--mode', choices=['random', 'augment'], default='random')
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--learning_rate', dest='lr', default=1e-4)
    parser.add_argument('--epoch', default=10)
    parser.add_argument('--add_num', default=20)
    parser.add_argument('--init_num', default=50)
    parser.add_argument('--batch_size', dest='bs', default=32)

    args = parser.parse_args()

    if not os.path.exists('results/'):
        os.makedirs('results/')

    main(args)