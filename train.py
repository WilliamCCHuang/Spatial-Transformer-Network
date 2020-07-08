import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from datasets import DistortedMNIST  #, MNISTAddition, CoLocalisationMNIST
from base_models import BaseCNNModel, BaseFCNModel, BaseSTN
from models import CNNModel, FCNModel, STModel
from utils import (
    save_model
)


def build_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', help='Experiment number', type=int, default=1)
    parser.add_argument('--task_type', default='DistortedMNIST')
    parser.add_argument('--model_name', default='ST-CNN')
    parser.add_argument('--transform_type', default='R')

    parser.add_argument('--img_size', default=28)
    parser.add_argument('--in_planes', default=1)
    parser.add_argument('--val_split', type=float, default=1 / 6)

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', default=0.01)

    parser.add_argument('--seed', type=int, default=42)

    return parser


def check_argparse(args):
    assert args.task_type in ['DistortedMNIST', 'MNISTAddition', 'CoLocalisationMNIST']
    assert args.transform_type in ['R', 'RTS', 'P', 'E', 'T', 'TU', None]
    assert args.model_name in ['CNN', 'FCN', 'ST-CNN', 'ST-FCN']


def build_train_val_test_dataset(args):
    if args.task_type == 'DistortedMNIST':
        train_dataset = DistortedMNIST(mode='train', transform_type=args.transform_type, val_split=args.val_split, seed=args.seed)
        val_dataset = DistortedMNIST(mode='val', transform_type=args.transform_type, val_split=args.val_split, seed=args.seed)
        test_dataset = DistortedMNIST(mode='test', transform_type=args.transform_type, seed=args.seed)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        return train_dataloader, val_dataloader, test_dataloader

    elif args.task_type == 'MNISTAddition':
        # TODO
        raise NotImplementedError
    else:
        # TODO
        raise NotImplementedError


def build_model(args):
    if args.task_type == 'DistortedMNIST':
        if args.model_name == 'ST-CNN':
            stn = BaseSTN(model_name=args.model_name, input_ch=args.in_planes, img_size=args.img_size)
            base_cnn = BaseCNNModel(input_length=args.input_length)
            model = STModel(base_stn=stn, base_nn_model=base_cnn)
        elif args.model_name == 'ST-FCN':
            stn = BaseSTN(model_name=args.model_name, in_planes=args.in_planes, img_size=args.img_size)
            base_fcn = BaseFCNModel(img_size=args.img_size)
            model = STModel(base_stn=stn, base_nn_model=base_fcn)
        elif args.model_name == 'CNN':
            model = CNNModel()
        else:
            model = FCNModel()
    elif args.task_type == 'MNISTAddition':
        raise NotImplementedError  # TODO
    else:
        raise NotImplementedError  # TODO
    
    return stn, model


def build_scheduler(optimizer):
    lr_lambda = lambda iteration: 0.1 ** (iteration // 50000)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    return scheduler


def main():
    # assert torch.cuda.is_available(), 'It is better to train with GPU'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # args
    parser = build_argparse()
    args = parser.parse_args()
    check_argparse(args)

    train_dataloader, val_dataloader, test_dataloader = build_train_val_test_dataset(args)

    stn, model = build_model(args)
    stn, model = stn.to(device), model.to(device)
            
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = build_scheduler(optimizer)

    writer = SummaryWriter(f'runs/trial_{args.exp}')

    for epoch in range(args.epochs):  # TODO: paper use 150*1000 iterations ~ 769 epoch in batch_size = 256
        train_running_loss = 0.0
        print(f'\n---The {epoch+1}-th epoch---\n')
        print('[Epoch, Batch] : Loss')

        print('---Training Loop begins---')

        for i, data in enumerate(train_dataloader):
            x_train, y_train = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_running_loss += loss.item()
            writer.add_scalar('Averaged loss', loss.item(), epoch * len(train_dataloader) + i)
            if i % 20 == 19:
                print(f'[{epoch+1}, {i+1}]: %.3f' % (train_running_loss / 20))
                train_running_loss = 0.0
            elif i == 195:
                print(f'[{epoch + 1}, {i + 1}]: %.3f' % (train_running_loss / 16))

            if i % 100:
                writer.add_scalar('norm', model.norm, i)

        print('---Training Loop ends---')
        
        with torch.no_grad():
            n = 6  # number of images to show
            original_img = x_train[:n, ...].clone().detach()  # (4, C, H, W)
            transformed_img = stn(original_img)  # (4, C, H, W)
            img = torch.cat((original_img, transformed_img), dim=0)  # (4 + 4, C, H, W)
            img = make_grid(img, nrow=n)
            writer.add_image(f'Original-Up, ST-Down images in epoch_{epoch+1}', img)
        
        print('---Validaion Loop begins---')

        with torch.no_grad():
            val_run_loss = 0.0
            batch_count = 0
            total_count = 0
            correct_count = 0
            for i, data in enumerate(val_dataloader, start=0):
                input, target = data[0].to(device), data[1].to(device)

                output = model(input)
                loss = criterion(output, target)

                _, predicted = torch.max(output, 1)

                val_run_loss += loss.item()
                batch_count += 1
                total_count += target.size(0)

                correct_count += (predicted == target).sum().item()
            
            accuracy = (100 * correct_count / total_count)
            val_run_loss = val_run_loss / batch_count
            
            writer.add_scalar('Validation accuracy', accuracy, epoch)
            writer.add_scalar('Validation loss', val_run_loss, epoch)

            print(f'Loss of {epoch+1} epoch is %.3f' % (val_run_loss))
            print(f'Accuracy is {accuracy} %')
                
            print('---Validaion Loop ends---')
    writer.close()

    print('\n-------- End Training --------\n')
    
    print('\n-------- Saving Model --------\n')

    save_model(model, args)
    
    print('\n-------- Saved --------\n')
    print(f'\n== Trial {args.exp} finished ==\n')

# ----------------------
# 10 epoch ~ 1m, accuracy 86.6%
# ----------------------


if __name__ == '__main__':
    main()
