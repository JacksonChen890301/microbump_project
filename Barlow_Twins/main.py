# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import numpy as np

from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from voxel_data_generator import CT_scan

parser = argparse.ArgumentParser(description='Barlow Twins Training')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--out-channel', default=1024, type=int, metavar='N',
                    help='output channel')
parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                    help='base learning rate for biases and batch norm parameters')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=0.01, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='2048-2048-2048', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=16, type=int, metavar='N',
                    help='print frequency')
os.environ['MASTER_ADDR']='localhost'
os.environ['MASTER_PORT']='5678'


def main():
    args, unknown = parser.parse_known_args()
    args.ngpus_per_node = torch.cuda.device_count()
    print(args.ngpus_per_node)

    torch.distributed.init_process_group(backend='nccl', rank=0, world_size=1)
    model = BarlowTwins(args)
    model = nn.DataParallel(model)
    model = model.cuda()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                     weight_decay_filter=True,
                     lars_adaptation_filter=True)
    # optimizer = optim.Adam(parameters, lr=1e-4)

    # automatically resume from checkpoint if it exists
    if Path('bumpSSL_checkpoint.pth').is_file():
        ckpt = torch.load('bumpSSL_checkpoint.pth', map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        start_epoch = 0

    dataset = CT_scan()
#     sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=True, shuffle=True)

    loss_list = []
    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):
        loss_per_epoch = 0.0
        for step, (y1, y2) in enumerate(loader, start=epoch * len(loader)):
            y1 = y1.cuda()
            y2 = y2.cuda()
            adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model.forward(y1, y2)
                loss_per_epoch += loss.mean().item()
            scaler.scale(loss.sum()).backward()
             
            if step % 8 == 0:    
                scaler.step(optimizer)
                scaler.update()
            if step % args.print_freq == 0:
                stats = dict(epoch=epoch, step=step,
                             lr_weights=optimizer.param_groups[0]['lr'],
                             lr_biases=optimizer.param_groups[1]['lr'],
                             loss=loss.sum().item(),
                             time=int(time.time() - start_time))
                print(stats)
                # save checkpoint
                state = dict(epoch=epoch + 1, model=model.state_dict(),
                             optimizer=optimizer.state_dict())
                torch.save(state, 'checkpoint_1007.pth')
        if epoch%100 == 0:
            torch.save(model.module.backbone.state_dict(), 'earlystop'+str(epoch)+'_B512_C1024.pth')
        loss_list.append(loss_per_epoch)

    # save final model
    loss_array = np.array(loss_list)
    np.save("BT_B512_C1024_loss.npy", loss_array)
    print(loss_list)
    torch.save(model.module.backbone.state_dict(),
               'BT_B512_C1024.pth')


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases


def handle_sigterm(signum, frame):
    pass


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def _weights_init(m):  # 權重初始化
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight)


class ConvNet_module(nn.Module):
    def __init__(self):
        super().__init__()
        base = 64
        self.conv1 = nn.Conv3d(1, base, kernel_size=3, stride=1, bias=False, padding=1)
        self.conv2 = nn.Conv3d(base, base*2, kernel_size=3, stride=1, bias=False, padding=1)
        self.conv3 = nn.Conv3d(base*2, base*4, kernel_size=3, stride=1, bias=False, padding=1)
        self.conv4 = nn.Conv3d(base*4, base*8, kernel_size=3, stride=1, bias=False, padding=1)
        self.conv5 = nn.Conv3d(base*8, base*16, kernel_size=3, stride=1, bias=False, padding=1)
        self.bn1 = nn.BatchNorm3d(base)
        self.bn2 = nn.BatchNorm3d(base*2)
        self.bn3 = nn.BatchNorm3d(base*4)
        self.bn4 = nn.BatchNorm3d(base*8)
        self.bn5 = nn.BatchNorm3d(base*16)
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)
        self.pool3 = nn.MaxPool3d(2)
        self.pool4 = nn.MaxPool3d(2)
        self.pool5 = nn.AvgPool3d(4)
        self.apply(_weights_init)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool5(x)
        x = torch.squeeze(x)
        return x


class BarlowTwins(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = ConvNet_module()
        self.backbone.fc = nn.Identity()

        # projector
        sizes = [args.out_channel] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):      
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)
        # c_rev = self.bn(z2).T @ self.bn(z1)
        
        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss


class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])

if __name__ == '__main__':
    main()
