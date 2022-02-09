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
    def __init__(self, num_classes=1, dim=32, num_tokens=8**3, token_c=64, mlp_dim=128, heads=8, depth=1, dropout=0.1):
        super(ViTNet, self).__init__()

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) (d p3)-> b (h w d) (p1 p2 p3 c)', p1=8, p2=8, p3=8),
            Linear(8**3, token_c),
        )

        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), token_c))
        torch.nn.init.normal_(self.pos_embedding, std=.02)  
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_c))  
        self.dropout = Dropout(dropout)

        self.transformer = Transformer(token_c, depth, heads, mlp_dim, dropout)
        self.to_cls_token = nn.Identity()
        # output
        self.nn1 = Linear(64, 1) 
        self.act1 = ReLU()
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)
        self.nn2 = Linear(8, num_classes)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std=1e-6)

    def forward(self, x, mask=None):
        T = self.to_patch_embedding(x)
        cls_tokens = self.cls_token.expand(T.shape[0], -1, -1)
        x = torch.cat((cls_tokens, T), dim=1)
        x += self.pos_embedding
        T = self.dropout(x)
        x = self.transformer(T)
        x = self.to_cls_token(x[:, 0])
#         x = self.act1(self.nn1(x))
        x = self.nn1(x)
        return x

    def relprop(self, cam=None, method="transformer_attribution", is_ablation=False, start_layer=0, **kwargs):
        cam = self.nn1.relprop(cam, **kwargs)
        cam = self.transformer.relprop(cam, **kwargs)
        if method == "transformer_attribution" or method == "grad":
            cams = []
            grad = self.transformer.attn.get_attn_gradients()
            cam = self.transformer.attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cams.append(cam.unsqueeze(0))
            rollout = compute_rollout_attention(cams, start_layer=start_layer)
            cam = rollout[:, 0, 1:]
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
    
    t_start = time.time()
    
#     lr = []
#     for step in range(1, 31):
#         lr.append(Noam_decay(step=step))
#     plt.plot(np.linspace(1, 30, 30), lr)
#     plt.xlabel('step')
#     plt.ylabel('lr')
#     plt.show()
#     plt.close()
    
    lr, epoch, min_val_loss = 1e-4, 100, 100
    tl, vl = [], []
    model = ViTNet().cuda()
    criterion = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    for i in range(epoch):
        train_loss, val_loss = 0.0, 0.0
        t, v = 0, 0
        for p in optim.param_groups:
            p['lr'] = Noam_decay(step=i+1)
        model.train()
        for inputs, labels in train_loader:
            y_pred = model(inputs)
            loss = criterion(y_pred, labels[:, 0].reshape(-1, 1))
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss += loss.item()
            t += 1
#         for inputs, labels in aug_loader:
#             y_pred = model(inputs)
#             loss = criterion(y_pred, , labels[:, 0].reshape(-1, 1))
#             optim.zero_grad()
#             loss.backward()
#             optim.step()
#             train_loss += loss.item()
#             t += 1
        model.eval()
        for inputs, labels in test_loader:
            y_pred = model(inputs)
            loss = criterion(y_pred, labels[:, 0].reshape(-1, 1))
            val_loss += loss.item()
            v += 1
        print('epoch: {}, Train Loss: {:.6f}, Eval Loss: {:.6f}'.format(i, train_loss/t, val_loss/v))
        tl.append(train_loss / t)
        vl.append(val_loss / v)
        if (val_loss / v) < min_val_loss:  # save the best model
            min_val_loss = (val_loss / v)
            torch.save(model, 'bumpViT.pth')
    
    t_end = time.time()
    print((t_end-t_start)/60)
    
    training_profile(epoch, tl, vl)
    
    model = torch.load('bumpViT.pth')
    model.eval()
    deviation = 0.0
    mean, sigma = 0.01411727, np.sqrt(7.34240751e-07)
    y, l = np.array([]), np.array([])
    for inputs, labels in test_loader:
            y_pred = model(inputs)
            tag = labels[:, 1]
            y_pred, labels = (y_pred*sigma)+mean, (labels[:, 0]*sigma)+mean
            y = np.append(y,y_pred.cpu().detach().numpy().reshape(-1, 1))
            l = np.append(l,labels.cpu().detach().numpy().reshape(-1, 1))
            for i in range(len(labels)):
                if tag[i] == 0:
                    plt.scatter(labels[i].cpu().detach().numpy(), y_pred[i,0].cpu().detach().numpy(), c='blue',alpha=0.5)
                else:
                    plt.scatter(labels[i].cpu().detach().numpy(), y_pred[i,0].cpu().detach().numpy(), c='red',alpha=0.5)
            deviation += torch.sum(torch.abs(y_pred[:,0]-labels)/labels)
    print(deviation/len(test_data))
    plt.plot([0.0125, 0.016], [0.0125, 0.016], c='black')
    mse = mean_squared_error(y, l)
    plt.text(0.0125, 0.0155, 'RMSE='+str(round(np.sqrt(mse), 4)),fontsize=12)
    plt.xlabel('ground truth')
    plt.ylabel('predition')
    plt.show()
    plt.close()
    error = np.abs((y-l)/l)
    plt.hist(error, bins=10)
    plt.ylim(0, 120)
    plt.xlabel('deviation')
    plt.ylabel('accumulation')
    plt.show()
    plt.close()

    testing_set(bump_test, model)
    
    img = test_data[124][0].cpu().detach().numpy()
    plt.imshow(img[0, :, :, 32], cmap='gray', vmax=np.max(img), vmin=np.min(img))
    plt.show()
    plt.close() 
    lrp = LRP(model)
    raw = test_data[124][0].unsqueeze(0)
    relevance_map(lrp, raw, model(raw))


def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    # all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
    #                       for i in range(len(all_layer_matrices))]
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer+1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)
    return joint_attention


def testing_set(dataset, model):
    test_data = bump_test()
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE_TRAIN, shuffle=False, num_workers=0)
    model.eval()
    deviation = 0.0
    mean, sigma = 0.01411727, np.sqrt(7.34240751e-07)
    y, l = np.array([]), np.array([])
    for inputs, labels in test_loader:
            y_pred = model(inputs)
            y_pred, labels = (y_pred*sigma)+mean, (labels*sigma)+mean
            y = np.append(y,y_pred.cpu().detach().numpy().reshape(-1, 1))
            l = np.append(l,labels.cpu().detach().numpy().reshape(-1, 1))
            plt.scatter(labels.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), c='red',alpha=0.5)
            deviation += torch.sum(torch.abs(y_pred-labels)/labels)
    print(deviation/len(test_data))
    plt.plot([0.0125, 0.0160], [0.0125, 0.0160], c='black')
    mse = mean_squared_error(y, l)
    plt.text(0.0125, 0.0155, 'RMSE='+str(round(np.sqrt(mse), 4)),fontsize=12)
    plt.ylabel('predition')
    plt.xlabel('ground truth')
    plt.show()
    plt.close()
    error = np.abs((y-l)/l)
    plt.hist(error, bins=10)
    plt.ylim(0, 18)
    plt.xlabel('deviation')
    plt.ylabel('accumulation')
    plt.show()
    plt.close()
    
    
def relevance_map(attribution_generator, original_image, class_index=None):
    transformer_attribution = attribution_generator.generate_LRP(original_image, method="transformer_attribution", output=1).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 8, 8, 8)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=8, mode='trilinear')
    transformer_attribution = transformer_attribution.reshape(64, 64, 64).cuda().data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.reshape(64, 64, 64).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    for i in range(64):
        vis = show_cam_on_image(image_transformer_attribution[:,:,i], transformer_attribution[:,:,i])
        vis =  np.uint8(255 * vis)
        vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
        plt.imshow(vis)
        plt.savefig('relevance_ViT/'+str(i+1))
        plt.show()
        plt.close()


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_HOT)
    heatmap = np.float32(heatmap) / 255
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cam = heatmap + np.float32(img)
    cam = cam / 2
    return cam

        
if __name__ == '__main__':
    main()
