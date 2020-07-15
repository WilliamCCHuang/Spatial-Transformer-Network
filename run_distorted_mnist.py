import os
import argparse
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import DistortedMNIST
from utils import error, save_model, load_best_model
from models import CNN, FCN, SpatialTransformer, SpatialTransformerNetwork


PATH = os.path.dirname(os.path.realpath(__file__))
LOGS_DIR = os.path.join(PATH, 'logs')
MODELS_DIR = os.path.join(PATH, 'models')

if not os.path.exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)
    
if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)


def build_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', help='Experiment number', type=int, default=1)
    parser.add_argument('--model_name', type=str, default='ST-CNN')
    parser.add_argument('--transform_type', type=str, default='R')

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--val_split', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)

    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    args.task_type = 'distorted_mnist'

    return args


def check_args(args):
    assert args.model_name in ['CNN', 'FCN', 'ST-CNN', 'ST-FCN']
    assert args.transform_type in ['R', 'RTS', 'P', 'E', 'T', 'TU', None]


def get_dataloaders(args):
    train_dataset = DistortedMNIST(mode='train', transform_type=args.transform_type, val_split=args.val_split, seed=args.seed)
    val_dataset = DistortedMNIST(mode='val', transform_type=args.transform_type, val_split=args.val_split, seed=args.seed)
    test_dataset = DistortedMNIST(mode='test', transform_type=args.transform_type, seed=args.seed)

    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


def get_model(model_name, img_size, in_channels):
    if model_name == 'ST-CNN':
        spatial_transformer = SpatialTransformer(model_name, img_size, in_channels, fc_units=1)
        backbone = CNN(img_size, in_channels)
        model = SpatialTransformerNetwork(spatial_transformer, backbone)
    elif model_name == 'ST-FCN':
        spatial_transformer = SpatialTransformer(model_name, img_size, in_channels, fc_units=1)
        backbone = FCN(img_size, in_channels)
        model = SpatialTransformerNetwork(spatial_transformer, backbone)
    elif model_name == 'CNN':
        model = CNN(img_size, in_channels)
    else:
        model = FCN(img_size, in_channels)

    return model


def get_scheduler(optimizer):
    lr_lambda = lambda iteration: 0.1 ** (iteration // 50000)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    return scheduler


def train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device, writer, args):
    steps = 0
    best_loss = np.inf
    best_error = np.inf

    for epoch in enumerate(tqdm(range(args.epochs), leave=False, desc='training')):
        model.train()

        for i, (imgs, labels) in enumerate(train_dataloader):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            train_loss = criterion(outputs, labels)

            train_loss.backward()
            optimizer.step()
            scheduler.step(i)

            writer.add_scalar('train loss', train_loss.item(), steps)

            if i % 100 == 0:
                writer.add_scalar('norm', model.norm, steps)

            steps += 1

        val_loss, val_error = evaluate(model, val_dataloader, criterion, writer, args)
        print('Epoch {2d}: val loss = {.4f}, val error = {.4f}'.format(epoch, val_loss, val_error))

        if best_loss > val_loss or best_error > val_error:
            best_loss = val_loss if best_loss > val_loss else best_loss
            best_error = val_error if best_error > val_error else best_error

            save_model(model, val_loss, val_error, args, MODELS_DIR)

    model = load_best_model(model, args, MODELS_DIR)

    return model


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, writer, args, epoch=None):
    model.eval()

    total = len(dataloader)
    classes = next(dataloader)[1].size(1)
    y_true = torch.zeros((total,))
    y_pred = torch.zeros((total, classes))

    for i, (imgs, labels) in enumerate(tqdm(dataloader, leave=False, desc='evaluating')):
        imgs, labels = imgs.to(device), labels.to(device)

        start = i * args.batch_size
        end = (i + 1) * args.batch_size

        outputs = model(imgs)

        y_pred[start:end] = outputs
        y_true[start:end] = labels

    assert end == len(dataloader), 'some data are left'

    loss = criterion(y_pred, y_true)
    y_pred = y_pred.argmax(dim=-1)
    error_rate = error(y_pred, y_true)
    
    loss_tag = 'val_loss' if epoch is not None else 'test_loss'
    error_tag = 'val_error' if epoch is not None else 'test_error'
    epoch = epoch or 0
    
    writer.add_scalar(loss_tag, loss, epoch)
    writer.add_scalar(error_tag, error_rate, epoch)

    return loss.item(), error_rate.item()


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    args = build_args()
    check_args(args)

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(args)
    in_channels, width, height = train_dataloader.dataset[0][0].size()

    model = get_model(args.model_name, width, in_channels)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = get_scheduler(optimizer)

    writer = SummaryWriter(os.path.join(LOGS_DIR, f'exp_{args.exp}'))

    model = train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device, writer, args)
    test_loss, test_error = evaluate(model, test_dataloader, criterion, device, writer, args)

    writer.close()


if __name__ == "__main__":
    main()
