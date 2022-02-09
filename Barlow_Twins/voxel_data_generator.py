# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import random
import time
from torch.utils.data import Dataset
from PIL import Image, ImageOps, ImageFilter
from matplotlib.pyplot import imshow, close, show
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# +
def normalize(volume):
    min = np.min(volume)
    max = np.max(volume)
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


class CT_scan(Dataset):
    def __init__(self, ):
        x1, x2 = np.load('SSL/bump_dataset/bump_x_train.npy'), np.load('SSL/bump_dataset/bump_x_val.npy')
        x = np.append(x1, x2, axis=0)
        del x1, x2
        self.x = np.stack((x*255,)*3, axis=-1)

    def __getitem__(self, index):
        inputs = self.x[index, :, :, :, :]
        augmentation = Transform()
        return augmentation(inputs)

    def __len__(self):
        return len(self.x)


class SSL_Dataset(Dataset):
    def __init__(self, train=True):
        self.train = train
        initial = np.load('SSL/bump_dataset/N17_initial.npy')
        reflow = np.load('SSL/bump_dataset/N17_reflow.npy')
        image = np.load('SSL/bump_dataset/N17_ini_raw.npy')
        image = normalize(image)
        
        deviation_list = []
        for row in range(initial.shape[0]):
            for column in range(initial.shape[1]):
                deviation = (reflow[row, column]-initial[row, column])
                if math.isnan(deviation):
                    continue
                deviation_list.append(deviation*100)
        scaler = StandardScaler()
        deviation_list = scaler.fit_transform(np.array(deviation_list).reshape(-1, 1))
        # sort_res = np.sort(deviation_list, axis=0)
        # p = 320//3
        # split_p1, split_p2 = sort_res[p, :], sort_res[p*2, :]
        
        # one_hot_res = np.zeros((320, 3))
        # for i in range(320):
        #     if deviation_list[i, 0] < split_p1:
        #         one_hot_res[i, :] = np.array([1, 0, 0])
        #     elif deviation_list[i, 0] > split_p2:
        #         one_hot_res[i, :] = np.array([0, 0, 1])
        #     else:
        #         one_hot_res[i, :] = np.array([0, 1, 0]) 
                
        self.x_train = image[:290, :, :, :, :]
        self.x_val = image[290:, :, :, :, :]
        self.y_train = deviation_list[:290, :]
        self.y_val = deviation_list[290:, :]
                
    def __getitem__(self, index):
        if self.train:
            inputs, target = self.x_train[index, :, :, :, :], self.y_train[index, :]
        else:
            inputs, target = self.x_val[index, :, :, :, :], self.y_val[index, :]
        return inputs, target
    
    def __len__(self):
        if self.train:
            return len(self.x_train)
        else:
            return len(self.x_val)
        
    
class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            axis = random.randint(0, 2)
            sigma = random.random() * 1.9 + 0.1
            if axis==0:
                for i in range(img.shape[1]):
                    raw = Image.fromarray(img[0, i, :, :, :].copy().astype(np.uint8))
                    img[0, i, :, :, :] = np.array(raw.filter(ImageFilter.GaussianBlur(sigma)))
            elif axis==1:
                for i in range(img.shape[2]):
                    raw = Image.fromarray(img[0, :, i, :, :].copy().astype(np.uint8))
                    img[0, :, i, :, :] = np.array(raw.filter(ImageFilter.GaussianBlur(sigma)))
            else:
                for i in range(img.shape[3]):
                    raw = Image.fromarray(img[0, :, :, i, :].copy().astype(np.uint8))
                    img[0, :, :, i, :] = np.array(raw.filter(ImageFilter.GaussianBlur(sigma)))
            return img
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            for i in range(img.shape[1]):
                raw = Image.fromarray(img[0, i, :, :, :].copy().astype(np.uint8))
                img[0, i, :, :, :] = np.array(ImageOps.solarize(raw))
            return img
        else:
            return img


class RandomFlip(object):
    def __init__(self, p):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            axis = random.randint(0, 2)
            fliped = np.flip(img[0, :, :, :, :].copy(), axis=axis)
            img[0, :, :, :, :] = fliped
            return img
        else:
            return img
        
        
class ColorDistortion(object):
    def __init__(self, p):
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            seed = random.randint(0, 99999)
            t = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            for i in range(img.shape[1]): 
                torch.manual_seed(seed)
                raw = Image.fromarray(img[0, i, :, :, :].copy().astype(np.uint8))
                img[0, i, :, :, :] = np.array(t.forward(raw))
            return img
        else:
            return img

        
class RandomCrop(object):
    def __init__(self, p):
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            x, y, z = random.randint(0, 31), random.randint(0, 31), random.randint(0, 31) 
            new_img = torch.from_numpy(img[:, x:x+32, y:y+32, z:z+32, :]).permute(0, 4, 1, 2, 3)
            img = torch.nn.functional.interpolate(new_img, scale_factor=2, mode='trilinear', align_corners=True)
            return img.permute(0, 2, 3, 4, 1).numpy()
        else:
            return img

    
class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            RandomCrop(p=0.5),
            RandomFlip(p=0.5),
            ColorDistortion(p=0.8),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
#             transforms.ToTensor()
        ])
        self.transform_prime = transforms.Compose([
            RandomCrop(p=0.5),
            RandomFlip(p=0.5),
            ColorDistortion(p=0.8),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
#             transforms.ToTensor()
        ])

    def __call__(self, x):
        y1 = self.transform(x.copy())
        y2 = self.transform_prime(x.copy())
        y1 = np.transpose(y1, (0, 4, 1, 2, 3))[:, 0, :, :, :]/255
        y2 = np.transpose(y2, (0, 4, 1, 2, 3))[:, 0, :, :, :]/255
        return y1, y2
    
# start_time = time.time()
# print(CT_scan()[0][0].shape)
# plt.imshow(CT_scan()[0][0][0, :, :, 32], cmap="gray", vmin=0, vmax=1)
# plt.show()
# plt.close()
# plt.imshow(CT_scan()[0][1][0, :, :, 32], cmap="gray", vmin=0, vmax=1)
# end_time = time.time()
# print(end_time-start_time)

        
