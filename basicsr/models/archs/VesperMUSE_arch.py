import os
import random
import numbers
import warnings
import math

import numpy as np
import cv2
from PIL import Image
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out
from einops import rearrange
import torchvision.transforms as T
from torchvision.models import (
    resnet18, resnet50, resnet101, resnet152,
    ResNet18_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
)

import albumentations as A
import kornia.augmentation as K
import kornia.filters as KF
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


# ==============================================================================
# Utils
# ==============================================================================
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
    
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


# ==============================================================================
# Attention Modules
# ==============================================================================
class MHSA(nn.Module):
    def __init__(self, dim, num_heads, bias=True):
        super(MHSA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.proj_q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.proj_kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        
        q = self.proj_q(x) 
        kv = self.proj_kv(x) 
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) 
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.proj_out(out)

class MHSA_Thermal(nn.Module):
    def __init__(self, dim, num_heads, bias=True):
        super(MHSA_Thermal, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x_rgb, x_thermal):
        """Cross-attention injection between RGB (main) and Thermal (aux)."""
        B, C, H, W = x_rgb.shape

        q = self.q_proj(x_rgb)
        k = self.k_proj(x_rgb)
        v = self.v_proj(x_thermal)

        q = rearrange(q, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)
        k = rearrange(k, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)
        v = rearrange(v, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)

        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h c (h1 w1) -> b (h c) h1 w1', h=self.num_heads, h1=H, w1=W)
        return self.out_proj(out)

class CrossModalMHSA(nn.Module):
    def __init__(self, dim, num_heads=8, bias=True):
        super(CrossModalMHSA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.proj_q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.proj_k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.proj_spatial = nn.Conv2d(2, 2 * dim, kernel_size=1, bias=bias)
        self.proj_global = nn.Conv1d(2, 2 * dim, kernel_size=1, bias=bias)
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x_rgb, x_thermal, x_depth):
        B, C, H, W = x_rgb.shape

        q = self.proj_q(x_rgb)
        k = self.proj_k(x_rgb)

        # Spatial MaxPool fusion
        t_pool = F.adaptive_max_pool2d(x_thermal, (H, W))
        d_pool = F.adaptive_max_pool2d(x_depth, (H, W))
        
        t_map = t_pool.max(dim=1, keepdim=True)[0]
        d_map = d_pool.max(dim=1, keepdim=True)[0]
        
        spatial_cat = torch.cat([t_map, d_map], dim=1)
        spatial_feat = self.proj_spatial(spatial_cat)
        spatial_feat = rearrange(spatial_feat, 'b c h w -> b (h w) c')

        # Global Average Pool fusion
        t_gap = F.adaptive_avg_pool2d(x_thermal, 1).view(B, C)
        d_gap = F.adaptive_avg_pool2d(x_depth, 1).view(B, C)
        
        global_cat = torch.stack([t_gap, d_gap], dim=1)
        global_feat = self.proj_global(global_cat)

        v = torch.matmul(spatial_feat, global_feat)
        v = v.permute(0, 2, 1)
        v = rearrange(v, 'b c (h w) -> b c h w', h=H, w=W)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)
        return self.proj_out(out)

class CrossModalMHSA_LLIE(nn.Module):
    def __init__(self, dim, num_heads=8, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.proj_q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.proj_k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.proj_spatial = nn.Conv2d(1, dim, kernel_size=1, bias=bias)
        self.proj_global = nn.Conv1d(1, dim, kernel_size=1, bias=bias)
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x_rgb, x_thermal):
        B, C, H, W = x_rgb.shape

        q = self.proj_q(x_rgb)
        k = self.proj_k(x_rgb)

        t_pool = F.adaptive_max_pool2d(x_thermal, (H, W))
        t_map = t_pool.max(dim=1, keepdim=True)[0]
        spatial_feat = self.proj_spatial(t_map)
        spatial_feat = rearrange(spatial_feat, 'b c h w -> b (h w) c')

        t_gap = F.adaptive_avg_pool2d(x_thermal, 1).view(B, C)
        global_feat = self.proj_global(t_gap.unsqueeze(1))

        v = torch.matmul(spatial_feat, global_feat)
        v = rearrange(v, 'b (h w) c -> b c h w', h=H, w=W)

        q = rearrange(q, 'b (h c) H W -> b h c (H W)', h=self.num_heads)
        k = rearrange(k, 'b (h c) H W -> b h c (H W)', h=self.num_heads)
        v = rearrange(v, 'b (h c) H W -> b h c (H W)', h=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h c (H W) -> b (h c) H W', H=H, W=W)
        return self.proj_out(out)

class FFN(nn.Module):
    def __init__(self, dim, mult=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x)


# ==============================================================================
# Transformer Blocks
# ==============================================================================
class TransformerBlock_LLIE(nn.Module):
    def __init__(self, in_channels, num_heads, dim_feedforward=None, LayerNorm_type='WithBias'):
        super(TransformerBlock_LLIE, self).__init__()
        if dim_feedforward is None:
            dim_feedforward = in_channels * 2

        self.cross_attention = MHSA_Thermal(in_channels, num_heads)
        self.norm1 = LayerNorm(dim=in_channels, LayerNorm_type=LayerNorm_type)
        self.ffn = FFN(dim=in_channels)
        self.norm2 = LayerNorm(dim=in_channels, LayerNorm_type=LayerNorm_type)

    def forward(self, x_rgb, x_thermal):
        attn_out = self.cross_attention(x_rgb, x_thermal)
        x = x_rgb + self.norm1(attn_out)
        x = x + self.norm2(self.ffn(x))
        return x

class TransformerBlock_Simple(nn.Module):
    def __init__(self, in_channels, num_heads, dim_feedforward=None, LayerNorm_type='WithBias'):
        super(TransformerBlock_Simple, self).__init__()
        if dim_feedforward is None:
            dim_feedforward = in_channels * 2

        self.attention = MHSA(in_channels, num_heads)
        self.norm1 = LayerNorm(dim=in_channels, LayerNorm_type=LayerNorm_type)
        self.ffn = FFN(dim=in_channels)
        self.norm2 = LayerNorm(dim=in_channels, LayerNorm_type=LayerNorm_type)

    def forward(self, x_rgb):
        attn_out = self.attention(x_rgb)
        x = x_rgb + self.norm1(attn_out)
        x = x + self.norm2(self.ffn(x))
        return x

class TransformerBlock_Fusion(nn.Module):
    def __init__(self, in_channels, num_heads, dim_feedforward=None, LayerNorm_type='WithBias'):
        super(TransformerBlock_Fusion, self).__init__()
        if dim_feedforward is None:
            dim_feedforward = in_channels * 2

        self.attention = CrossModalMHSA(in_channels, num_heads)
        self.norm1 = LayerNorm(dim=in_channels, LayerNorm_type=LayerNorm_type)
        self.ffn = FFN(dim=in_channels)
        self.norm2 = LayerNorm(dim=in_channels, LayerNorm_type=LayerNorm_type)

    def forward(self, x_rgb, x_thermal, x_depth):
        attn_out = self.attention(x_rgb, x_thermal, x_depth)
        x = x_rgb + self.norm1(attn_out)
        x = x + self.norm2(self.ffn(x))
        return x


# ==============================================================================
# Model Architecture
# ==============================================================================
class LLIE_Enhancer(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, n_feat=32, num_heads=2):
        super(LLIE_Enhancer, self).__init__()

        self.conv_in = nn.Conv2d(3, n_feat, kernel_size=1, padding="same")
        self.thermal_convs = nn.ModuleList([
            nn.Conv2d(1, n_feat, kernel_size=3, padding=1),
            nn.Conv2d(1, n_feat*2, kernel_size=3, padding=1),
            nn.Conv2d(1, n_feat*4, kernel_size=3, padding=1),
            nn.Conv2d(1, n_feat*2, kernel_size=3, padding=1),
            nn.Conv2d(1, n_feat, kernel_size=3, padding=1),
        ])

        # Encoder
        self.transformer_block1_1 = TransformerBlock_LLIE(n_feat, num_heads)
        self.down1 = nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=2, padding=1)
        self.transformer_block2_1 = TransformerBlock_LLIE(n_feat*2, num_heads)
        self.transformer_block2_2 = TransformerBlock_LLIE(n_feat*2, num_heads)
        self.down2 = nn.Conv2d(n_feat*2, n_feat*4, kernel_size=3, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = TransformerBlock_LLIE(n_feat*4, num_heads*2)

        # Decoder
        self.up2 = nn.ConvTranspose2d(n_feat*4, n_feat*2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.channel_adjust2 = nn.Conv2d(n_feat*4, n_feat*2, kernel_size=1)
        self.transformer_block_up2_1 = TransformerBlock_LLIE(n_feat*2, num_heads)

        self.up1 = nn.ConvTranspose2d(n_feat*2, n_feat, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.channel_adjust1 = nn.Conv2d(n_feat*2, n_feat, kernel_size=1)
        self.transformer_block_up1_1 = TransformerBlock_LLIE(n_feat, num_heads)

        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=1, padding="same")

    def forward(self, x):
        x_rgb = x[:, :3, :, :]
        x_thermal = x[:, 3:, :, :]

        t1 = self.thermal_convs[0](x_thermal)
        t2 = self.thermal_convs[1](F.interpolate(x_thermal, scale_factor=0.5, mode='bilinear', align_corners=False))
        t3 = self.thermal_convs[2](F.interpolate(x_thermal, scale_factor=0.25, mode='bilinear', align_corners=False))
        t4 = self.thermal_convs[3](F.interpolate(x_thermal, scale_factor=0.5, mode='bilinear', align_corners=False))
        t5 = self.thermal_convs[4](x_thermal)

        x = self.conv_in(x_rgb)

        x1 = self.transformer_block1_1(x, t1)
        x1_down = self.down1(x1)

        x2 = self.transformer_block2_1(x1_down, t2)
        x2 = self.transformer_block2_2(x2, t2)
        x2_down = self.down2(x2)

        bn = self.bottleneck(x2_down, t3)

        x2_up = self.up2(bn)
        x2_up = torch.cat([x2_up, x2], dim=1)
        x2_up = self.channel_adjust2(x2_up)
        x2_up = self.transformer_block_up2_1(x2_up, t4)

        x1_up = self.up1(x2_up)
        x1_up = torch.cat([x1_up, x1], dim=1)
        x1_up = self.channel_adjust1(x1_up)
        x1_up = self.transformer_block_up1_1(x1_up, t5)

        return self.conv_out(x1_up)

class FusionModule(nn.Module):
    def __init__(self, num_classes=1, main_variant="resnet50", aux_variant="resnet18", freeze_main=False):
        super(FusionModule, self).__init__()

        self.rgb_net = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.thm_net = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.dep_net = resnet18(weights=ResNet18_Weights.DEFAULT) 

        def adapt_first_layer(net):
            old_conv = net.conv1
            new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                new_conv.weight.data = old_conv.weight.data.sum(dim=1, keepdim=True)
            net.conv1 = new_conv
            return net

        self.thm_net = adapt_first_layer(self.thm_net)
        
        self.stem_r = nn.Sequential(self.rgb_net.conv1, self.rgb_net.bn1, self.rgb_net.relu, self.rgb_net.maxpool)
        self.layer1_r = self.rgb_net.layer1
        self.layer2_r = self.rgb_net.layer2
        self.layer3_r = self.rgb_net.layer3
        self.layer4_r = self.rgb_net.layer4

        self.stem_t = nn.Sequential(self.thm_net.conv1, self.thm_net.bn1, self.thm_net.relu, self.thm_net.maxpool)
        self.layer1_t = self.thm_net.layer1
        self.layer2_t = self.thm_net.layer2
        self.layer3_t = self.thm_net.layer3
        self.layer4_t = self.thm_net.layer4

        self.stem_d = nn.Sequential(self.dep_net.conv1, self.dep_net.bn1, self.dep_net.relu, self.dep_net.maxpool)
        self.layer1_d = self.dep_net.layer1
        self.layer2_d = self.dep_net.layer2
        self.layer3_d = self.dep_net.layer3
        self.layer4_d = self.dep_net.layer4

        dims_main = [256, 512, 1024, 2048] 
        dims_aux  = [64, 128, 256, 512]     

        self.proj_t1 = nn.Conv2d(dims_aux[0], dims_main[0], 1, bias=False)
        self.proj_t2 = nn.Conv2d(dims_aux[1], dims_main[1], 1, bias=False)
        self.proj_t3 = nn.Conv2d(dims_aux[2], dims_main[2], 1, bias=False)
        self.proj_t4 = nn.Conv2d(dims_aux[3], dims_main[3], 1, bias=False)

        self.proj_d1 = nn.Conv2d(dims_aux[0], dims_main[0], 1, bias=False)
        self.proj_d2 = nn.Conv2d(dims_aux[1], dims_main[1], 1, bias=False)
        self.proj_d3 = nn.Conv2d(dims_aux[2], dims_main[2], 1, bias=False)
        self.proj_d4 = nn.Conv2d(dims_aux[3], dims_main[3], 1, bias=False)

        self.cross_trans_1 = TransformerBlock_Fusion(dims_main[0], num_heads=4)
        self.cross_trans_2 = TransformerBlock_Fusion(dims_main[1], num_heads=4)
        self.cross_trans_3 = TransformerBlock_Fusion(dims_main[2], num_heads=8)
        self.cross_trans_4 = TransformerBlock_Fusion(dims_main[3], num_heads=16)

        self.up3 = nn.ConvTranspose2d(2048, 2048, 3, 2, 1, 1)
        self.channel_adjust3 = nn.Conv2d(2048 + 1024, 1024, 1)
        self.transformer_block_up3_1 = TransformerBlock_Simple(1024, num_heads=8)

        self.up2 = nn.ConvTranspose2d(1024, 1024, 3, 2, 1, 1)
        self.channel_adjust2 = nn.Conv2d(1024 + 512, 512, 1)
        self.transformer_block_up2_1 = TransformerBlock_Simple(512, num_heads=4)

        self.up1 = nn.ConvTranspose2d(512, 512, 3, 2, 1, 1)
        self.channel_adjust1 = nn.Conv2d(512 + 256, 256, 1)
        self.transformer_block_up1_1 = TransformerBlock_Simple(256, num_heads=4)

        self.aux_head1 = nn.Conv2d(1024, num_classes, 1)
        self.aux_head2 = nn.Conv2d(512, num_classes, 1)

        def freeze_bn(module):
            for m in module.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False
        
        for module in [self.stem_r, self.layer1_r, self.layer2_r, self.layer3_r, self.layer4_r,
                       self.stem_t, self.layer1_t, self.layer2_t, self.layer3_t, self.layer4_t,
                       self.stem_d, self.layer1_d, self.layer2_d, self.layer3_d, self.layer4_d]:
            freeze_bn(module)

    def forward(self, x_rgb, x_t, x_normals):
        H, W = x_rgb.shape[2], x_rgb.shape[3]

        f_r = self.stem_r(x_rgb)
        f_t = self.stem_t(x_t)
        f_d = self.stem_d(x_normals)

        f1_r = self.layer1_r(f_r)
        f1_t = self.layer1_t(f_t)
        f1_d = self.layer1_d(f_d)
        del f_r, f_t, f_d 

        f_f = self.cross_trans_1(f1_r, self.proj_t1(f1_t), self.proj_d1(f1_d))
        f1_r = f1_r + f_f
        skip1 = f_f 
        del f_f

        f2_r = self.layer2_r(f1_r)
        f2_t = self.layer2_t(f1_t)
        f2_d = self.layer2_d(f1_d)
        del f1_t, f1_d

        f_f = self.cross_trans_2(f2_r, self.proj_t2(f2_t), self.proj_d2(f2_d))
        f2_r = f2_r + f_f
        skip2 = f_f
        del f_f

        f3_r = self.layer3_r(f2_r)
        f3_t = self.layer3_t(f2_t)
        f3_d = self.layer3_d(f2_d)
        del f2_t, f2_d

        f_f = self.cross_trans_3(f3_r, self.proj_t3(f3_t), self.proj_d3(f3_d))
        f3_r = f3_r + f_f
        skip3 = f_f
        del f_f

        f4_r = self.layer4_r(f3_r)
        f4_t = self.layer4_t(f3_t)
        f4_d = self.layer4_d(f3_d)
        del f3_t, f3_d

        f4_final = self.cross_trans_4(f4_r, self.proj_t4(f4_t), self.proj_d4(f4_d))
        del f4_r, f4_t, f4_d

        x = self.up3(f4_final)
        if x.shape[-2:] != skip3.shape[-2:]:
            x = F.interpolate(x, size=skip3.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip3], dim=1)
        del skip3 
        x = self.channel_adjust3(x)
        stage3 = self.transformer_block_up3_1(x)

        x = self.up2(stage3)
        if x.shape[-2:] != skip2.shape[-2:]:
            x = F.interpolate(x, size=skip2.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip2], dim=1)
        del skip2
        x = self.channel_adjust2(x)
        stage2 = self.transformer_block_up2_1(x)

        x = self.up1(stage2)
        if x.shape[-2:] != skip1.shape[-2:]:
            x = F.interpolate(x, size=skip1.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip1], dim=1)
        del skip1
        x = self.channel_adjust1(x)
        x = self.transformer_block_up1_1(x)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

        aux = {"stage3": self.aux_head1(stage3), "stage2": self.aux_head2(stage2)}
        return x, aux

class VesperMUSE_General(nn.Module):
    def __init__(self, num_classes=1, use_llie=False, llie_path=None, depth_variant="depth-anything/Depth-Anything-V2-Small-hf", **kwargs):
        super(VesperMUSE_General, self).__init__()

        self.fusion_module = FusionModule(num_classes=num_classes)
        
        self.use_llie = use_llie
        if self.use_llie:
            self.llie_enhancer = LLIE_Enhancer()
            self._load_weights(self.llie_enhancer, llie_path)
            self.llie_enhancer.eval()
            for p in self.llie_enhancer.parameters():
                p.requires_grad = False

        self.depth_processor = AutoImageProcessor.from_pretrained(depth_variant)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(depth_variant)
        self.depth_model.eval()
        for p in self.depth_model.parameters():
            p.requires_grad = False

        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _load_weights(self, model, path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint.get('params', checkpoint)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
        model.to(device)

    def _depth_to_normals(self, depth):
        """Computes surface normals from normalized depth map."""
        gradients = KF.spatial_gradient(depth, order=1) 
        
        dx = gradients[:, :, 0, :, :]
        dy = gradients[:, :, 1, :, :]
        z_component = torch.ones_like(depth)
        
        normal_vec = torch.cat((-dx, -dy, z_component), dim=1)
        normal_vec = F.normalize(normal_vec, dim=1, p=2)
        
        normal_vis = (normal_vec + 1) / 2.0
        return normal_vis

    def forward(self, x):
        rgb = x[:, :3, :, :]
        thermal = x[:, 3:, :, :] 

        if self.use_llie:
            if not any(p.requires_grad for p in self.llie_enhancer.parameters()):
                with torch.no_grad():
                    rgb = self.llie_enhancer(torch.cat([rgb, thermal], dim=1))
            else:
                rgb = self.llie_enhancer(torch.cat([rgb, thermal], dim=1))
        
        rgb = rgb.clamp(0, 1) 
        thermal = thermal.clamp(0, 1)

        with torch.no_grad():
            h, w = rgb.shape[2], rgb.shape[3]
            
            depth_input = (rgb - torch.tensor([0.485, 0.456, 0.406], device=rgb.device).view(1,3,1,1)) / \
                          torch.tensor([0.229, 0.224, 0.225], device=rgb.device).view(1,3,1,1)
            
            depth_out = self.depth_model(depth_input).predicted_depth
            depth = depth_out.unsqueeze(1) 

            if depth.shape[-2:] != (h, w):
                depth = F.interpolate(depth, size=(h, w), mode='bilinear', align_corners=False)

            d_min = depth.amin(dim=(2, 3), keepdim=True)
            d_max = depth.amax(dim=(2, 3), keepdim=True)
            depth = (depth - d_min) / (d_max - d_min + 1e-8)

        normals = self._depth_to_normals(depth) 
        rgb_norm = self.normalize(rgb)
        normals_norm = self.normalize(normals)

        seg_out, aux = self.fusion_module(rgb_norm, thermal, normals_norm)
        return seg_out, aux

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SegmentationHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(32, num_classes, 1)
        )

    def forward(self, x):
        return self.conv(x)

class VesperMUSE_General_Seg(nn.Module):
    def __init__(self, num_classes=1, use_llie=False, llie_path=None):
        super(VesperMUSE_General_Seg, self).__init__()
        self.backbone = VesperMUSE_General(num_classes=num_classes, use_llie=use_llie, llie_path=llie_path)
        self.head = SegmentationHead(in_channels=256, num_classes=num_classes)

    def forward(self, x):
        fused, aux_data = self.backbone(x)
        logits = self.head(fused)
        return logits, aux_data

class RGB_Plain_SEG(nn.Module):
    def __init__(self, num_classes=5):
        super(RGB_Plain_SEG, self).__init__()

        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        self.stem = nn.Sequential(
            self.backbone.conv1, 
            self.backbone.bn1, 
            self.backbone.relu, 
            self.backbone.maxpool
        )
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4

        self.reduce4 = nn.Conv2d(2048, 256, 1)
        self.reduce3 = nn.Conv2d(1024, 256, 1)
        self.reduce2 = nn.Conv2d(512, 256, 1) 
        self.reduce1 = nn.Conv2d(256, 256, 1) 

        self.fuse3_reduce = nn.Conv2d(512, 256, 1)
        self.fuse3_trans = TransformerBlock_Simple(in_channels=256, num_heads=4)

        self.fuse2_reduce = nn.Conv2d(512, 128, 1)
        self.fuse2_trans = TransformerBlock_Simple(in_channels=128, num_heads=4)

        self.fuse1_reduce = nn.Conv2d(128 + 256, 64, 1)
        self.fuse1_trans = TransformerBlock_Simple(in_channels=64, num_heads=2)

        self.head = SegmentationHead(in_channels=64, num_classes=num_classes)

    def forward(self, x):
        x = x[:, :3, :, :]
        H, W = x.shape[2], x.shape[3]

        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        p4 = self.reduce4(f4)
        p4_up = F.interpolate(p4, size=f3.shape[2:], mode='bilinear', align_corners=False)
        p3 = self.reduce3(f3)
        
        cat3 = torch.cat([p4_up, p3], dim=1)
        out3 = self.fuse3_reduce(cat3)
        out3 = self.fuse3_trans(out3)

        out3_up = F.interpolate(out3, size=f2.shape[2:], mode='bilinear', align_corners=False)
        p2 = self.reduce2(f2)
        
        cat2 = torch.cat([out3_up, p2], dim=1)
        out2 = self.fuse2_reduce(cat2)
        out2 = self.fuse2_trans(out2)

        out2_up = F.interpolate(out2, size=f1.shape[2:], mode='bilinear', align_corners=False)
        p1 = self.reduce1(f1)
        
        cat1 = torch.cat([out2_up, p1], dim=1)
        out1 = self.fuse1_reduce(cat1)
        out1 = self.fuse1_trans(out1)

        features_full = F.interpolate(out1, size=(H, W), mode='bilinear', align_corners=False)
        return self.head(features_full)

class RGB_Plain_SEG_LLIE(nn.Module):
    def __init__(self, num_classes=5, weights_llie="/home/upt/VesperMuse/experiments/CART_LLIE/CART_LLIE/best_psnr_27.85_299500.pth"):
        super(RGB_Plain_SEG_LLIE, self).__init__()

        self.enhancer = LLIE_Enhancer()
        ckpt_llie = torch.load(weights_llie, map_location='cpu')
        
        state_dict = ckpt_llie.get('params', ckpt_llie)
        try:
            self.enhancer.load_state_dict(state_dict)
        except RuntimeError:
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.enhancer.load_state_dict(new_state_dict)
            
        self.enhancer.eval()
        for param in self.enhancer.parameters():
            param.requires_grad = False

        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        self.stem = nn.Sequential(
            self.backbone.conv1, 
            self.backbone.bn1, 
            self.backbone.relu, 
            self.backbone.maxpool
        )
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4

        self.reduce4 = nn.Conv2d(2048, 256, 1)
        self.reduce3 = nn.Conv2d(1024, 256, 1)
        self.reduce2 = nn.Conv2d(512, 256, 1)
        self.reduce1 = nn.Conv2d(256, 256, 1)

        self.fuse3_reduce = nn.Conv2d(512, 256, 1)
        self.fuse3_trans = TransformerBlock_Simple(in_channels=256, num_heads=4)

        self.fuse2_reduce = nn.Conv2d(512, 128, 1)
        self.fuse2_trans = TransformerBlock_Simple(in_channels=128, num_heads=4)

        self.fuse1_reduce = nn.Conv2d(128 + 256, 64, 1)
        self.fuse1_trans = TransformerBlock_Simple(in_channels=64, num_heads=2)

        self.head = SegmentationHead(in_channels=64, num_classes=num_classes)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]

        with torch.no_grad():
            x = self.enhancer(x)
            if isinstance(x, (list, tuple)):
                x = x[0]

        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        p4 = self.reduce4(f4)
        p4_up = F.interpolate(p4, size=f3.shape[2:], mode='bilinear', align_corners=False)
        p3 = self.reduce3(f3)
        out3 = self.fuse3_trans(self.fuse3_reduce(torch.cat([p4_up, p3], dim=1)))

        out3_up = F.interpolate(out3, size=f2.shape[2:], mode='bilinear', align_corners=False)
        p2 = self.reduce2(f2)
        out2 = self.fuse2_trans(self.fuse2_reduce(torch.cat([out3_up, p2], dim=1)))

        out2_up = F.interpolate(out2, size=f1.shape[2:], mode='bilinear', align_corners=False)
        p1 = self.reduce1(f1)
        out1 = self.fuse1_trans(self.fuse1_reduce(torch.cat([out2_up, p1], dim=1)))

        features_full = F.interpolate(out1, size=(H, W), mode='bilinear', align_corners=False)
        return self.head(features_full)