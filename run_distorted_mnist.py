import argparse

import torch
from torch.utils.data import DataLoader

from models import (
    BaseSTN,
    BaseCNNModel,
    BaseFCNModel,
    STModel,
    CNNModel,
    FCNModel
)
from datasets import DistortedMNIST


def build_args():
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

    args = parser.parse_args()

    return args


def check_args(args):
    assert args.task_type in ['DistortedMNIST', 'MNISTAddition', 'CoLocalisationMNIST']
    assert args.transform_type in ['R', 'RTS', 'P', 'E', 'T', 'TU', None]
    assert args.model_name in ['CNN', 'FCN', 'ST-CNN', 'ST-FCN']


def prepare_dataloaders(args):
    if args.task_type == 'DistortedMNIST':
        train_dataset = DistortedMNIST(mode='train', transform_type=args.transform_type, val_split=args.val_split, seed=args.seed)
        val_dataset = DistortedMNIST(mode='val', transform_type=args.transform_type, val_split=args.val_split, seed=args.seed)
        test_dataset = DistortedMNIST(mode='test', transform_type=args.transform_type, seed=args.seed)
    elif args.task_type == 'MNISTAddition':
        raise NotImplementedError  # TODO:
    else:
        raise NotImplementedError  # TODO:

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


def build_model(args):
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

    return model


def build_scheduler():
    pass  # TODO:


def train():
    pass  # TODO:


@torch.no_grad()
def evaluate():
    pass  # TODO:


def main():
    pass  # TODO:


if __name__ == "__main__":
    main()
