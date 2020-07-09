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


def build_scheduler(optimizer):
    lr_lambda = lambda iteration: 0.1 ** (iteration // 50000)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    return scheduler


def main():
    # assert torch.cuda.is_available(), 'It is better to train with GPU'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # args
    parser = build_argparse()
    
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
