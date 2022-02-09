from pathlib import Path
import argparse
import json
import os
import random
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

from torch import nn, optim
from torchvision import models, datasets, transforms
import torch
import torchvision
import torch.nn.functional as F
from voxel_data_generator import SSL_Dataset

parser = argparse.ArgumentParser(description='Evaluate')
parser.add_argument('--weights', default='freeze', type=str,
                    choices=('finetune', 'freeze'),
                    help='finetune or freeze resnet weights')
parser.add_argument('--train-percent', default=100, type=int,
                    choices=(100, 10, 1),
                    help='size of traing set in percent')
parser.add_argument('--workers', default=0, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr-backbone', default=0.0, type=float, metavar='LR',
                    help='backbone base learning rate')
parser.add_argument('--lr-classifier', default=0.3, type=float, metavar='LR',
                    help='classifier base learning rate')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--print-freq', default=5, type=int, metavar='N',
                    help='print frequency')


class ConvNet_module(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=1, bias=False, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=1, bias=False, padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, stride=1, bias=False, padding=1)
        self.conv4 = nn.Conv3d(256, 512, kernel_size=3, stride=1, bias=False, padding=1)
        self.conv5 = nn.Conv3d(512, 1024, kernel_size=3, stride=1, bias=False, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(256)
        self.bn4 = nn.BatchNorm3d(512)
        self.bn5 = nn.BatchNorm3d(1024)
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)
        self.pool3 = nn.MaxPool3d(2)
        self.pool4 = nn.MaxPool3d(2)
        self.pool5 = nn.AvgPool3d(4)
        self.fc = nn.Linear(1024,256)
        self.out = nn.Linear(256, 1)
#         self.sf = nn.Softmax(dim=1)
        
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
        x = self.out(F.relu(self.fc(x)))
        return x


def main():
    # result_analysis()
    args, unknown = parser.parse_known_args() 
    args.ngpus_per_node = torch.cuda.device_count()
    model = ConvNet_module().cuda()
    state_dict = torch.load('earlystop400_B512_C1024.pth', map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
#     assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()
    model.out.weight.data.normal_(mean=0.0, std=0.01)
    model.out.bias.data.zero_()
    if args.weights == 'freeze':
        model.requires_grad_(False)
        model.fc.requires_grad_(True)
    classifier_parameters, model_parameters = [], []
    for name, param in model.named_parameters():
        if name in {'fc.weight', 'fc.bias'}:
            classifier_parameters.append(param)
        else:
            model_parameters.append(param)
            
    # different losses
#     criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()

    param_groups = [dict(params=classifier_parameters, lr=args.lr_classifier)]
    if args.weights == 'finetune':
        param_groups.append(dict(params=model_parameters, lr=args.lr_backbone))
    optimizer = optim.Adam(param_groups, lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Data loading
    train_dataset = SSL_Dataset(train=True)
    val_dataset = SSL_Dataset(train=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers)

    start_time = time.time()
    best_val = 10
    loss_list, valloss_list = [], []
    for epoch in range(args.epochs):
        loss_per_epoch, valloss_per_epoch = 0.0, 0.0
        # train
        if args.weights == 'finetune':
            model.train()
        elif args.weights == 'freeze':
            model.eval()
        else:
            assert False
#         correct, total = torch.zeros(1).squeeze().cuda(), torch.zeros(1).squeeze().cuda()
        for step, (images, target) in enumerate(train_loader, start=epoch * len(train_loader)):
            output = model(images.cuda(non_blocking=True))
            loss = criterion(output, target.cuda(non_blocking=True).float())
            loss_per_epoch += loss.item()
            # for classification
#             loss = criterion(output, target.cuda(non_blocking=True).squeeze().long())
#             correct, total = acc_cal(output, target, correct, total)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % args.print_freq == 0:
                pg = optimizer.param_groups
                stats = dict(epoch=epoch, step=step, loss=loss.item(),
                             time=int(time.time() - start_time))
                print(stats)
                for inputs, labels in val_loader:
                    y_pred = model(inputs.cuda())
                    loss = criterion(y_pred, labels.cuda(non_blocking=True).float())
#                     loss = criterion(y_pred, labels.cuda(non_blocking=True).squeeze().long())
                    valloss_per_epoch = loss.item()
                    print('val_loss', valloss_per_epoch)
                    if loss.item() < best_val:
                        best_val = loss.item()
                        torch.save(model.state_dict(),'Sup_B512_C1024.pth')
        loss_list.append(loss_per_epoch)
        valloss_list.append(valloss_per_epoch)
#         scheduler.step()
#         print("Accuracy = ", (correct/total).cpu().detach().numpy())
        
        # sanity check
#         if args.weights == 'freeze':
#             reference_state_dict = torch.load('../bumpSSL_earlystop.pth', map_location='cpu')
#             model_state_dict = model.state_dict()
#             for k in reference_state_dict:
#                 assert torch.equal(model_state_dict[k].cpu(), reference_state_dict[k]), k
       
    print(best_val)
    # loss_profile(loss_list, valloss_list)
#     loss_array = np.array(loss_list)
#     np.save("Sup_B1024_C1024_1000loss.npy", loss_array)
    state = torch.load('Sup_B512_C1024.pth', map_location='cpu')
    model.load_state_dict(state, strict=False)
    val_set(val_loader, model)

    
def val_set(dataset, model):
    test_loader = dataset
    model.eval()
    deviation = 0.0
    mean, sigma = 0, 1
    y, l = np.array([]), np.array([])
    s = 0
    for inputs, labels in test_loader:
            y_pred = model(inputs.cuda())
            y_pred, labels = ((y_pred.cpu().detach().numpy().reshape(-1, 1)*sigma)+mean), ((labels.cpu().detach().numpy().reshape(-1, 1)*sigma)+mean)
            y = np.append(y, y_pred)
            l = np.append(l, labels.reshape(-1, 1))
            plt.scatter(labels, y_pred, c='red',alpha=0.5)
    plt.plot([-2, 2.2], [-2, 2.2], c='black', ls='--')
#     plt.text(0.0125/1.5*1000, 0.0155/1.5*1000, 'RMSE='+str(round(np.sqrt(mse), 4))+'(mΩ)',fontsize=12)
    plt.ylabel('predition', fontsize=18)
    plt.xlabel('ground truth', fontsize=18)
    plt.show()
    plt.close()
    
    
def acc_cal(output, labels, correct, total):
    prediction = torch.argmax(output, 1)
    correct += (prediction == labels.cuda().squeeze()).sum().float()
    total += len(labels)
    return correct, total


def loss_profile(profile=None, profile_val=None):
#     profile = np.load('BT_B1024_C1024_loss.npy')
    plt.title('BT_B1024_C1024', fontsize=18)
    plt.plot(np.arange(500), profile, c='blue', label='train')
    plt.plot(np.arange(500), profile_val, c='red', label='val')
    plt.xlabel('epoch', fontsize=18)
    plt.ylabel('loss', fontsize=18)
    plt.show()
    plt.close()
    
    
def result_analysis():
    loss_list = [0.66, 0.52, 0.47, 0.42, 0.4]
    loss_1024 = [0.39]
    plt.plot([600, 700, 800, 900, 1000], loss_list, c='red', marker='o')
    plt.title('B512_C1024', fontsize=18)
    plt.xlabel('BT training epoch', fontsize=18)
    plt.ylabel('MSE loss(resistance)', fontsize=18)
    plt.savefig('mse.png')
    plt.close()


if __name__ == '__main__':
    main()