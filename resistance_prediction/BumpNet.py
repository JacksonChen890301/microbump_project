# -*- coding: utf-8 -*-
# +
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.init as init
import os
import time
import cv2
import netron
from layers_LRP import *
from microbump_generator import CT_scan, bump_test
from voxel_data_generator import bump_augmentation
from transformer import Transformer
from projector import Projector
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
from einops.layers.torch import Rearrange
from sklearn.metrics import mean_squared_error
from VT_explanation_generator import LRP


# GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
# torch.manual_seed(0)
BATCH_SIZE_TRAIN = 64


def _weights_init(m):  # 權重初始化
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)


def Noam_decay(step, warmup=5):  # learning rate 先升後降
    lr = 64**(-1.5)*min([step**(-0.5), warmup**(-1.5)*step])
    return lr


class ViTNet(nn.Module):
    def __init__(self, num_classes=1, dim=32, num_tokens=8, token_c=32, mlp_dim=128, heads=8, depth=2, dropout=0.1):
        super(ViTNet, self).__init__()

        self.conv1 = Conv3d(1, 16, kernel_size=3, stride=1, bias=False, padding=1)
        self.conv2 = Conv3d(16, dim, kernel_size=3, stride=1, bias=False, padding=1)
        self.bn1 = BatchNorm3d(16)
        self.bn2 = BatchNorm3d(dim)
        self.pool1 = MaxPool3d(2)
        self.pool2 = MaxPool3d(2)
        self.apply(_weights_init)  # 權重初始化

        # Tokenization
        self.token_wA = Linear(dim, num_tokens)  # Tokenization parameters
        torch.nn.init.xavier_uniform_(self.token_wA.weight)
        self.mat = einsum('bij,bjk->bik')
        self.token_wV = Linear(dim, token_c)  # Tokenization parameters
        torch.nn.init.xavier_uniform_(self.token_wV.weight)

        self.transformer = Transformer(token_c, depth, heads, mlp_dim, dropout)
        self.to_cls_token = nn.Identity()
        self.projector = Projector(in_channels=dim, out_channels=16, token_channels=token_c)

        # output
        self.nn1 = Linear(16*16*16*16, 2048) 
        self.act1 = ReLU()
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)
        self.do1 = Dropout(dropout)
        self.nn2 = Linear(2048, 128)
        self.act2 = ReLU()
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std=1e-6)
        self.nn3 = Linear(128, num_classes)
        torch.nn.init.xavier_uniform_(self.nn3.weight)
        torch.nn.init.normal_(self.nn3.bias, std=1e-6)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = rearrange(x, 'b c d h w -> b (d h w) c')

        # Tokenization
        A = self.token_wA(x)  
        A = rearrange(A, 'b h w -> b w h') 
        A = A.softmax(dim=-1)
        T = self.mat([A, x])
        T = self.token_wV(T) 

        T = self.transformer(T)  
        x = self.projector(x, T)
        x = rearrange(x, 'b s c -> b (s c)')
        x = self.act1(self.nn1(x))
        x = self.do1(x)
        x = self.act2(self.nn2(x))
        x = self.nn3(x)
        return x

    def relprop(self, cam=None, method="transformer_attribution", mode=0, start_layer=0, **kwargs):
        cam = self.nn3.relprop(cam, **kwargs)
        cam = self.act2.relprop(cam, **kwargs)
        cam = self.nn2.relprop(cam, **kwargs)
        cam = self.act1.relprop(cam, **kwargs)
        cam = self.nn1.relprop(cam, **kwargs)
        cam = cam.reshape(1, 4096, 16)
        cam = self.projector.relprop(cam, mode,**kwargs)
        cam = self.transformer.relprop(cam, **kwargs)
        cam = self.token_wV.relprop(cam, **kwargs)
        (cam1, cam2) = self.mat.relprop(cam, **kwargs)
        if mode == 0:
            cam = cam1.transpose(1, 2)
            cam = self.token_wA.relprop(cam, **kwargs)
        else:
            cam = cam2
        cam = rearrange(cam, 'b (d h w) c -> b c d h w', d=16, h=16, w=16)
        cam = self.pool2.relprop(cam, **kwargs)
        cam = self.bn2.relprop(cam, **kwargs)
        cam = self.conv2.relprop(cam, **kwargs)
        cam = self.pool1.relprop(cam, **kwargs)
        cam = self.bn1.relprop(cam, **kwargs)
        cam = self.conv1.relprop(cam, **kwargs)
        return cam

    
def training_profile(epoch, train_loss, val_loss):
    plt.plot(np.linspace(1, epoch, epoch), train_loss, c='green', label='train')
    plt.plot(np.linspace(1, epoch, epoch), val_loss, c='red', label='validate')
    plt.ylim([0, 1])
    plt.xlabel('epoch')
    plt.ylabel('loss(MSE)')
    plt.legend()
    plt.show()
    plt.close()
    
    
def main():
    # data loading
    train_data = CT_scan(train=True)
    aug_data = bump_augmentation(train=True)
    test_data = CT_scan(train=False)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=0)
    aug_loader = DataLoader(dataset=aug_data, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE_TRAIN, shuffle=False, num_workers=0)
    print(len(train_data), len(test_data))
    
#     t_start = time.time()
    
#     lr = []
#     for step in range(1, 51):
#         lr.append(Noam_decay(step=step))
#     plt.plot(np.linspace(1, 50, 50), lr)
#     plt.xlabel('step')
#     plt.ylabel('lr')
#     plt.show()
#     plt.close()
    
#     lr, epoch, min_val_loss = 1e-4, 150, 100
#     tl, vl = [], []
#     model = ViTNet().cuda()
#     criterion = torch.nn.MSELoss()
#     optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=3e-3)
#     for i in range(epoch):
#         train_loss, val_loss = 0.0, 0.0
#         t, v = 0, 0
#         for p in optim.param_groups:
#             p['lr'] = Noam_decay(step=i+1)
#         model.train()
#         for inputs, labels in train_loader:
#             y_pred = model(inputs)
#             loss = criterion(y_pred, labels[:, 0].reshape(-1, 1))
#             optim.zero_grad()
#             loss.backward()
#             optim.step()
#             train_loss += loss.item()
#             t += 1
#         for inputs, labels in aug_loader:
#             y_pred = model(inputs)
#             loss = criterion(y_pred, labels[:, 0].reshape(-1, 1))
#             optim.zero_grad()
#             loss.backward()
#             optim.step()
#             train_loss += loss.item()
#             t += 1
#         model.eval()
#         for inputs, labels in test_loader:
#             y_pred = model(inputs)
#             loss = criterion(y_pred, labels[:, 0].reshape(-1, 1))
#             val_loss += loss.item()
#             v += 1
#         print('epoch: {}, Train Loss: {:.6f}, Eval Loss: {:.6f}'.format(i, train_loss/t, val_loss/v))
#         tl.append(train_loss / t)
#         vl.append(val_loss / v)
#         if (val_loss / v) < min_val_loss:  # save the best model
#             min_val_loss = (val_loss / v)
#             torch.save(model, 'bump_hybrid_aug.pth')
    
#     t_end = time.time()
#     print((t_end-t_start)/60)
    
#     training_profile(epoch, tl, vl)
    
    model = torch.load('bump_hybrid_aug.pth')
    model.eval()
    deviation = 0.0
    mean, sigma = 0.01411727, np.sqrt(7.34240751e-07)
    y, l = np.array([]), np.array([])
    s1, s2 = 0, 0
    for inputs, labels in test_loader:
            y_pred = model(inputs)
            tag = labels[:, 1]
            y_pred, labels = ((y_pred*sigma)+mean)/1.5*1000, ((labels[:, 0]*sigma)+mean)/1.5*1000
            y = np.append(y,y_pred.cpu().detach().numpy().reshape(-1, 1))
            l = np.append(l,labels.cpu().detach().numpy().reshape(-1, 1))
            for i in range(len(labels)):
                if tag[i] == 0:
                    s1 = plt.scatter(labels[i].cpu().detach().numpy(), y_pred[i,0].cpu().detach().numpy(), c='blue',alpha=0.5)
                else:
                    s2 = plt.scatter(labels[i].cpu().detach().numpy(), y_pred[i,0].cpu().detach().numpy(), c='red',alpha=0.5)
            deviation += torch.sum(torch.abs(y_pred[:,0]-labels)/labels)
    print(deviation/len(test_data))
    plt.plot([0.0125/1.5*1000, 0.0165/1.5*1000], [0.0125/1.5*1000, 0.0165/1.5*1000], c='black', ls='--')
    print(np.corrcoef(y, l))
    mse = mean_squared_error(y, l)
    plt.text(0.0125/1.5*1000, 0.0155/1.5*1000, 'RMSE='+str(round(np.sqrt(mse), 2))+'(mΩ)',fontsize=12)
    plt.xlabel('ground truth(mΩ)', fontsize=18)
    plt.ylabel('predition(mΩ)', fontsize=18)
    plt.legend(handles=[s1, s2], labels=['initial', 'reflow'], loc='lower right')
    plt.show()
    plt.close()

    testing_set(bump_test, model)

#     img = test_data[124][0].cpu().detach().numpy() # 185, 229, 124, 167/142, 117, 297,  40
#     np.save('sample124.npy', img)
#     plt.imshow(img[0, 32, :, :], cmap='gray', vmax=1.1, vmin=np.min(img))
#     plt.axis('off')
#     plt.show()
#     plt.close() 
#     weight = weight_vis(model, test_data[167][0].reshape(1, 1, 64, 64, 64)).cpu().detach().numpy().reshape(1, 16, 16, 16, 16)
#     for i in range(8):
#         for j in range(16):
#             print(j)
#             plt.imshow(weight[0, i, :, :, j], cmap='hot', vmax=np.max(weight[0, i, :, :, :]), vmin=np.min(weight[0, i, :, :, :]))
#             plt.axis('off')
# #             plt.savefig('attention/'+str(i+1)+'_'+str(j+1))
#             plt.show()
#             plt.close()
    lrp = LRP(model)
    raw = test_data[124][0].unsqueeze(0)
    relevance_map(lrp, raw, 10)


def testing_set(dataset, model):
    test_data = bump_test()
    print(len(test_data))
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE_TRAIN, shuffle=False, num_workers=0)
    model.eval()
    deviation = 0.0
    mean, sigma = 0.01411727, np.sqrt(7.34240751e-07)
    y, l = np.array([]), np.array([])
    s = 0
    for inputs, labels in test_loader:
            y_pred = model(inputs)
            y_pred, labels = ((y_pred*sigma)+mean)/1.5*1000, ((labels*sigma)+mean)/1.5*1000
            y = np.append(y,y_pred.cpu().detach().numpy().reshape(-1, 1))
            l = np.append(l,labels.cpu().detach().numpy().reshape(-1, 1))
            s = plt.scatter(labels.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), c='red',alpha=0.5)
            deviation += torch.sum(torch.abs(y_pred-labels)/labels)
    print(deviation/len(test_data))
    plt.plot([0.0125/1.5*1000, 0.0165/1.5*1000], [0.0125/1.5*1000, 0.0165/1.5*1000], c='black', ls='--')
    print(np.corrcoef(y, l))
    mse = mean_squared_error(y, l)
    plt.text(0.0125/1.5*1000, 0.0155/1.5*1000, 'RMSE='+str(round(np.sqrt(mse), 4))+'(mΩ)',fontsize=12)
    plt.ylabel('predition(mΩ)', fontsize=18)
    plt.xlabel('ground truth(mΩ)', fontsize=18)
    plt.legend(handles=[s], labels=['reflow'], loc='lower right')
    plt.show()
    plt.close()

def relevance_map(attribution_generator, original_image, class_index=None):
    transformer_attribution = attribution_generator.generate_LRP(original_image, method="transformer_attribution",mode=1, output=1).detach()
    transformer_attribution = transformer_attribution.reshape(64, 64, 64).cuda().data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.reshape(64, 64, 64).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = np.empty((64, 64, 64, 3))
    for i in range(64):
        vis[i, :, :, :] = show_cam_on_image(image_transformer_attribution[:,:,i], transformer_attribution[:,:,i])
    vis = vis/vis.max()
    for i in range(64):
        v = vis[i, :, :, :]
        v =  np.uint8(255 * v)
        v = cv2.cvtColor(np.array(v), cv2.COLOR_RGB2BGR)
        plt.imshow(v)
        plt.axis('off')
        plt.savefig('relevance_VT/'+str(i+1))
        plt.show()
        plt.close()


def show_cam_on_image(img, mask):
#     mask[mask<0.1] = 0
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_HOT)
    heatmap = np.float32(heatmap) / 255
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cam = heatmap + np.float32(img)
    return cam


def weight_vis(net, x):
    x = F.relu(net.bn1(net.conv1(x)))
    x = net.pool1(x)
    x = F.relu(net.bn2(net.conv2(x)))
    x = net.pool2(x)
    x = rearrange(x, 'b c d h w -> b (d h w) c')

    # Tokenization
    A = net.token_wA(x)  
    A = rearrange(A, 'b h w -> b w h') 
    A = A.softmax(dim=-1)
    T = net.mat([A, x])
    T = net.token_wV(T) 
    T = net.transformer(T)  
    x = net.projector(x, T)
    return x


if __name__ == '__main__':
    main()
# -


