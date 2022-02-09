# +
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange
from einops.layers.torch import Rearrange
from layers_LRP import *


class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.nn1 = Linear(dim, hidden_dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)
        self.af1 = nn.GELU()
        self.do1 = nn.Dropout(dropout)
        self.nn2 = Linear(hidden_dim, dim)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std=1e-6)
        self.do2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.nn1(x)
        x = self.af1(x)
        x = self.do1(x)
        x = self.nn2(x)
        x = self.do2(x)
        return x

    def relprop(self, cam, **kwargs):
        cam = self.nn2.relprop(cam, **kwargs)
        cam = self.nn1.relprop(cam, **kwargs)
        return cam


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector, thats why *3
        self.to_qk = Linear(dim, dim * 2, bias=True)
        torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        torch.nn.init.zeros_(self.to_qkv.bias)
        self.softmax = Softmax(dim=-1)
        self.attn_drop = Dropout(dropout)
#         # A = Q*K^T
#         self.matmul1 = einsum('bhid,bhjd->bhij')
#         # attn = A*V
#         self.matmul2 = einsum('bhij,bhjd->bhid')
        
        self.matmul1 = einsum('bij,bjk->bik')
        self.matmul2 = einsum('bij,bjk->bik')

        self.nn1 = Linear(dim, dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.zeros_(self.nn1.bias)
        self.do1 = Dropout(dropout)

        self.attn_cam = None
        self.attn = None
        self.v = None
        self.v_cam = None
        self.attn_gradients = None
        self.clone1 = Clone()

#     def forward(self, x):
#         b, n, _, h = *x.shape, self.heads
#         qkv = self.to_qkv(x)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
#         q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)  # split into multi head attentions
#         self.save_v(v)
#         dots = self.matmul1([q, k]) * self.scale
#         attn = self.softmax(dots)  # follow the softmax,q,d,v equation in the paper
#         attn = self.attn_drop(attn)
#         self.save_attn(attn)
#         attn.register_hook(self.save_attn_gradients)
#         out = self.matmul2([attn, v])  
#         out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block
#         out = self.nn1(out)
#         out = self.do1(out)
#         return out
    
    def forward(self, x):
        x, t = self.clone1(x, 2)
        b, n, _, h = *x.shape, self.heads
        qk = self.to_qk(x)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k = rearrange(qk, 'b n (qk d) -> qk b n d', qk=2)  # split into multi head attentions
        q = rearrange(q, 'b t c -> b c t')
        dots = self.matmul1([k, q]) * self.scale
        attn = self.softmax(dots)  # follow the softmax,q,d,v equation in the paper
        attn = self.attn_drop(attn)
        self.save_attn(attn)
        attn.register_hook(self.save_attn_gradients)  
        out = self.matmul2([attn, t]) 
        return out

    def relprop(self, cam, **kwargs):
        (cam1, cam_v)= self.matmul2.relprop(cam, **kwargs)
        print(torch.min(cam1), torch.min(cam_v))
        cam1 /= 2
        cam_v /= 2
        self.save_v_cam(cam_v)
        self.save_attn_cam(cam1)
        # A = Q*K^T
        (cam_k, cam_q) = self.matmul1.relprop(cam1, **kwargs)
        cam_q = rearrange(cam_q, 'b c t -> b t c')
        cam_q /= 2
        cam_k /= 2
        cam_qk = rearrange([cam_q, cam_k], 'qk b n d -> b n (qk d)', qk=2)
        cam = self.to_qk.relprop(cam_qk, **kwargs)
        cam = self.clone1.relprop((cam, cam_v), **kwargs)
        return cam

    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn

    def save_attn_cam(self, cam):
        self.attn_cam = cam

    def get_attn_cam(self):
        return self.attn_cam

    def get_v(self):
        return self.v

    def save_v(self, v):
        self.v = v

    def save_v_cam(self, cam):
        self.v_cam = cam

    def get_v_cam(self):
        return self.v_cam

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads=heads, dropout=dropout)
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.mlp = MLP_Block(dim, mlp_dim, dropout=dropout)

        self.add1 = Add()
        self.add2 = Add()
        self.clone1 = Clone()
        self.clone2 = Clone()

    def forward(self, x):
        x1, x2 = self.clone1(x, 2)
        x = self.add1([x1, self.attn(self.norm1(x2))])
        x1, x2 = self.clone2(x, 2)
        x = self.add2([x1, self.mlp(self.norm2(x2))])
        return x

    def relprop(self, cam, **kwargs):
        (cam1, cam2) = self.add2.relprop(cam, **kwargs)
        cam2 = self.mlp.relprop(cam2, **kwargs)
        cam2 = self.norm2.relprop(cam2, **kwargs)
        cam = self.clone2.relprop((cam1, cam2), **kwargs)

        (cam1, cam2) = self.add1.relprop(cam, **kwargs)
        cam2 = self.attn.relprop(cam2, **kwargs)
        cam2 = self.norm1.relprop(cam2, **kwargs)
        cam = self.clone1.relprop((cam1, cam2), **kwargs)
        return cam
    
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim=dim, heads=heads, mlp_dim=mlp_dim, dropout=dropout)
            for i in range(depth)])
        
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x
    
    def relprop(self, cam, **kwargs):
        for blk in reversed(self.blocks):
            cam = blk.relprop(cam, **kwargs)
        return cam

