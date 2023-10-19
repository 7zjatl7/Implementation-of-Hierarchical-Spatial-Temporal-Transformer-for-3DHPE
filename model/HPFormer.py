import sys
sys.path.insert(0, '../')
import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import time

from math import sqrt
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from common.utils import *
from model.module.trans import Block
from model.module.trans import SinusoidalPositionEmbeddings


class HPFormer(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None, is_train=True, args=None):

        super().__init__()
        self.args = args
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        in_dim = 3
        out_dim = 3     #### output dimension is num_joints * 3
        head_embed_dim = embed_dim * 4
        
        self.is_train=is_train
        self.body_part_set = [[1, 2, 3], 
                              [4, 5, 6], 
                              [0, 7, 8], 
                              [11, 12, 13], 
                              [14, 15, 16],
                              [9, 10]]
        
        ### spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.Body_pos_temporal_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.Pos_temporal_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim))
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.block_depth = depth

        self.STEblocks = nn.ModuleList([
            # Block: Attention Block
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.TTEblocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, comb=False, changedim=False, currentdim=i+1, depth=depth)
            for i in range(depth)])

        self.BTTEblocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, comb=False, changedim=False, currentdim=i+1, depth=depth)
            for i in range(depth)])
        
        self.PTTEblocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, comb=False, changedim=False, currentdim=i+1, depth=depth)
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)
        
        self.feature_forward = nn.Sequential(
            nn.LayerNorm(head_embed_dim),
            nn.Linear(head_embed_dim , embed_dim),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , out_dim)
        )


    def STE_forward(self, x):
        b, f, n, c = x.shape  ##### b is batch size, f is number of frames, n is number of joints, c is channel size?
        
        x = rearrange(x, 'b f n c  -> (b f) n c', )
        ### now x is [batch_size, receptive frames, joint_num, 2 channels]
        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed

        x = self.pos_drop(x)
        for blk in self.STEblocks:
            x = blk(x)
            # x = blk(x, vis=True)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) n c -> (b n) f c', f=f)
        
        return x

    def TTE_forward(self, x):
        assert len(x.shape) == 3, "shape is equal to 3"
        b, f, _  = x.shape
 
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        for blk in self.TTEblocks:
            x = blk(x)
        # x = blk(x, vis=True)
        # exit()

        x = self.Temporal_norm(x)
        return x

    def BTTE_forward(self, x):
        # assert len(x.shape) == 3, "shape is equal to 3"
        b, f, n, c  = x.shape
            
        # Composition of a set of body part set
        body_set = []
        bs_blk = []
        blk = self.BTTEblocks[0]
        x_b = torch.rand([b, f, n, c]).cuda()
        
        for bp in self.body_part_set:
            bp_x = x[:, :, bp, :]
            bp_x = rearrange(bp_x, 'b f n c -> (b n) f c')
            body_set.append(bp_x)
        
        # extract temporal feature through body_part_set 
        for i, bs in enumerate(body_set):
            # for blk in self.BTTEblocks:
            bs += self.Body_pos_temporal_embed
            temporal_bs = blk(bs)
            bp_x = rearrange(temporal_bs, '(b n) f c -> b f n c', n=len(self.body_part_set[i]))
            bs_blk.append(bp_x)
        
        # arrange order of pose
        for i, bp in enumerate(self.body_part_set):
            x_b[:,:,bp,:] = bs_blk[i]
        x = self.Temporal_norm(x_b)
        
        return x

    def PTTE_forward(self, x):
        assert len(x.shape)==4, "shape is equal to 4"
        b, f, n, c = x.shape
        
        x += self.Pos_temporal_embed
        x = rearrange(x, 'b f n c -> b (f n) c')
        x = self.pos_drop(x)
        # blk = self.PTTEblocks[0]
        for blk in self.PTTEblocks:
            x = blk(x)
        x = rearrange(x, 'b (f n) c -> b f n c', f=f, n=n)
        
        return x
    

    def forward(self, x):
        b, f, n, c = x.shape
        x = self.STE_forward(x)
        x_ste = rearrange(x, '(b n) f c -> b f n c', n=n)
        x = self.TTE_forward(x)
        x_tte = rearrange(x, '(b n) f c -> b f n c', n=n)
        x = rearrange(x, '(b n) f c -> b f n c', n=n)
        x = self.BTTE_forward(x)
        x_btte = x.clone()
        x = self.PTTE_forward(x)
        x_ptte = x.clone()
        # 행방향으로 cat fc레이어로 4D x D를 곱해줘서 T x J x D로 변경
        x = torch.cat([x_ste, x_tte, x_btte, x_ptte], dim=-1)
        x = self.feature_forward(x)
        # 그리고 head로 D -> 3으로 변경
        x = self.head(x)
        return x
    