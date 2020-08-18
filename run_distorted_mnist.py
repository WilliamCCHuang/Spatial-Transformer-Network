import os
import argparse
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from datasets import DistortedMNIST
from utils import error, save_model, load_model
from models import CNN, FCN, SpatialTransformerModule, SpatialTransformerNetwork


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

    parser.add_argument('--iterations', type=int, default=15 * 10**4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--val_split', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_decay_steps', type=int, default=50000)
    parser.add_argument('--lr_decay_gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--recode_norm', type=int, default=1000)
    parser.add_argument('--record_image', type=int, default=1000)

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    args.task_type = 'distorted_mnist'

    return args


def check_args(args):
    assert args.model_name in ['CNN', 'FCN', 'ST-CNN', 'ST-FCN']
    assert args.transform_type in ['R', 'RTS', 'P', 'E', 'T', 'TU', None]
    
    if (args.device not in ['cpu', 'tpu']) and ('cuda' not in args.device):
        raise RuntimeError('`--device` can only be `cpu`, `cuda`, `tpu`')


def get_device(args):
    if args.device == 'tpu':
        if not os.environ['COLAB_TPU_ADDR']:
            raise RuntimeError('Make sure to select TPU from Edit > Notebook settings > Hardware accelerator')
        
        VERSION = '20200325'  # @param ["1.5" , "20200325", "nightly"]
        os.system('curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py')
        os.system(f'python pytorch-xla-env-setup.py --version {VERSION}')

        # import torch_xla
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
    elif args.device == 'cpu':
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device(args.device)
        else:
            raise RuntimeError('Not found GPU!')

    print(f'Using device: {device}')

    return device


def get_dataloaders(args):
    train_dataset = DistortedMNIST(mode='train', transform_type=args.transform_type, val_split=args.val_split, seed=args.seed)
    val_dataset = DistortedMNIST(mode='val', transform_type=args.transform_type, val_split=args.val_split, seed=args.seed)
    test_dataset = DistortedMNIST(mode='test', transform_type=args.transform_type, seed=args.seed)

    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, args.batch_size, shuffle=False, )
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


def get_model(model_name, img_size, in_channels):
    if model_name == 'ST-CNN':
        spatial_transformer = SpatialTransformerModule(model_name, img_size, in_channels, fc_units=6)
        backbone = CNN(img_size, in_channels)
        model = SpatialTransformerNetwork(spatial_transformer, backbone)
    elif model_name == 'ST-FCN':
        spatial_transformer = SpatialTransformerModule(model_name, img_size, in_channels, fc_units=6)
        backbone = FCN(img_size, in_channels)
        model = SpatialTransformerNetwork(spatial_transformer, backbone)
    elif model_name == 'CNN':
        model = CNN(img_size, in_channels)
    else:
        model = FCN(img_size, in_channels)

    return model


def get_scheduler(optimizer, args):
    lr_lambda = lambda iteration: args.lr_decay_gamma ** (iteration // args.lr_decay_steps)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    return scheduler


def train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device, writer, args):
    total_data = len(train_dataloader.dataset)
    iterations_per_epoch = np.floor(total_data / args.batch_size)
    epochs = int(np.floor(args.iterations / iterations_per_epoch))

    steps = 0
    stop = False
    best_loss = np.inf
    best_error = np.inf

    for epoch in tqdm(range(epochs), leave=False, desc='training'):
        model.train()

        for imgs, labels in train_dataloader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            train_loss = criterion(outputs, labels)

            train_loss.backward()
            optimizer.step()
            scheduler.step(steps)

            writer.add_scalar('train loss', train_loss.item(), steps)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], steps)

            if steps % args.recode_norm == 0:
                writer.add_scalar('norm', model.norm, steps)
            
            if steps % args.record_image == 0:
                ori_imgs = imgs[:10]  # (10, 1, W, H)
                aug_imgs = model.transform(ori_imgs)  # (10, 1, H, W)
                all_imgs = torch.cat((ori_imgs, aug_imgs), dim=0)  # (20, 1, H, W)
                all_imgs = make_grid(all_imgs, nrow=10)  # (3, H, W)

                writer.add_image('Original / Transformed', all_imgs, steps)
            
            steps += 1
            
            if steps == args.iterations:
                stop = True
                break

        val_loss, val_error = evaluate(model, val_dataloader, criterion, device, writer, args, epoch)
        print('Epoch {:2d}: val loss = {:.4f}, val error = {:.4f}'.format(epoch, val_loss, val_error))

        if best_loss > val_loss or best_error > val_error:
            best_loss = val_loss if best_loss > val_loss else best_loss
            best_error = val_error if best_error > val_error else best_error

            save_model(model, args, MODELS_DIR)

        if stop:
            print('training done!')
            break

    model = load_model(model, args, MODELS_DIR)

    return model


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, writer, args, epoch=None):
    model.eval()

    total = len(dataloader.dataset)
    classes = 10

    y_true = torch.zeros((total,), dtype=torch.long)
    y_pred = torch.zeros((total, classes), dtype=torch.float)

    for i, (imgs, labels) in enumerate(dataloader):
        imgs, labels = imgs.to(device), labels.to(device)

        start = i * args.batch_size
        end = min((i + 1) * args.batch_size, total)

        outputs = model(imgs)

        y_pred[start:end, :] = outputs
        y_true[start:end] = labels

    assert end == total, 'some data are left'

    loss = criterion(y_pred, y_true)
    y_pred = y_pred.argmax(dim=-1)
    error_rate = error(y_pred, y_true)
    
    loss_tag = 'val_loss' if epoch is not None else 'test_loss'
    error_tag = 'val_error' if epoch is not None else 'test_error'
    epoch = epoch or -1
    
    writer.add_scalar(loss_tag, loss, epoch)
    writer.add_scalar(error_tag, error_rate, epoch)

    return loss.item(), error_rate


def main():
    args = build_args()
    check_args(args)

    device = get_device(args)
    
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(args)
    in_channels, width, height = train_dataloader.dataset[0][0].size()

    model = get_model(args.model_name, width, in_channels)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = get_scheduler(optimizer)

    writer = SummaryWriter(os.path.join(LOGS_DIR, f'exp_{args.exp}'))

    model = train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device, writer, args)
    test_loss, test_error = evaluate(model, test_dataloader, criterion, device, writer, args)

    print('Final: test loss = {:.4f}, test error = {:.4f}'.format(test_loss, test_error))

    writer.close()


if __name__ == "__main__":
    main()
