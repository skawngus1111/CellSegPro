import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

def argparsing() :
    parser = argparse.ArgumentParser(
        description='Following are the arguments that can be passed form the terminal itself!')
    parser.add_argument('--data_path', type=str,
                        default='/media/jhnam19960514/68334fe0-2b83-45d6-98e3-76904bf08127/home/namjuhyeon/Desktop/LAB/common material/Dataset Collection')
    parser.add_argument('--data_type', type=str, default='data-science-bowl-2018')
    parser.add_argument('--model_name', type=str, default='fcn8s')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--save_path', type=str, default='./model_save')
    parser.add_argument('--pretrained_encoder', default=False, action='store_true')
    parser.add_argument('--train', default=False, action='store_true')

    # Train parameter
    parser.add_argument('--k_fold', type=int, default=5)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--criterion', type=str, default='BCE')
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-2)

    # Print parameter
    parser.add_argument('--step', type=int, default=10)

    args = parser.parse_args()

    return args

def get_deivce() :
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("You are using \"{}\" device.".format(device))

    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_model(device, model_name, num_classes, num_channels, canny,
              image_size=256, angle=20, length=20, preserve_range=10, num_mask=4) :
    if model_name == 'fcn8s' :
        from models.fcn import fcn8s
        return fcn8s(num_classes)
    elif model_name == 'deeplabv3+' :
        from models.deeplabv3plus import DeepLabv3_plus
        return DeepLabv3_plus(nInputChannels=1, n_classes=1)
    elif model_name == 'unet' :
        from models.unet import Unet
        return Unet(1, 1, 64)
    elif model_name == 'fdsnet' :
        from models.FDSNet.fdsnet import FDSNet
        return FDSNet(device, image_size, angle, length, preserve_range, num_mask, canny)
    else:
        print("{} does not be implemented...".format(model_name))
        sys.exit(1)

def get_criterion(criterion) :
    if criterion == 'CCE' :
        criterion = nn.CrossEntropyLoss()
    elif criterion == 'BCE' :
        criterion = nn.BCEWithLogitsLoss()
    else :
        print("Wrong criterion")
        sys.exit()

    return criterion

def get_optimizer(optimizer, model, lr, momentum, weight_decay):
    params = list(filter(lambda p: p.requires_grad, model.parameters()))

    if optimizer == 'SGD' :
        optimizer = optim.SGD(params=params, lr=lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    elif optimizer == 'Adam' :
        optimizer = optim.Adam(params=params, lr=lr, weight_decay=weight_decay)
    elif optimizer == 'AdamW' :
        optimizer = optim.AdamW(params=params, lr=lr, weight_decay=weight_decay)
    else :
        print("Wrong optimizer")
        sys.exit()

    return optimizer

def get_lr(step, total_steps, lr_max, lr_min):
  """Compute learning rate according to cosine annealing schedule."""
  return lr_min + (lr_max - lr_min) * 0.5 * (1 +
                                             np.cos(step / total_steps * np.pi))

def get_scheduler(optimizer, epochs, train_loader_len, learning_rate) :
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
            step,
            epochs * train_loader_len,
            1,  # lr_lambda computes multiplicative factor
            1e-6 / learning_rate))

    return scheduler

def get_save_path(**kwargs) :
    model_dir = os.path.join(kwargs['save_path'], kwargs['data_type'])
    model_dirs = os.path.join(model_dir,
                              str(kwargs['image_size']),
                              str(kwargs['batch_size']),
                              kwargs['model_name'])

    if 'fold' in kwargs.keys() :
        model_dirs = os.path.join(model_dirs, 'fold {}'.format(kwargs['fold'] + 1))

    if not os.path.exists(model_dirs): os.makedirs(model_dirs)

    save_model_path = '{}_{}'.format(kwargs['lr'], str(kwargs['epochs']).zfill(3))

    return model_dirs, save_model_path