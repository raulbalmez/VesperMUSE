import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from pdb import set_trace as stx
# import cv2


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


# input [bs,28,256,310]  output [bs, 28, 256, 256]
def shift_back(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    down_sample = 256 // row
    step = float(step) / float(down_sample * down_sample)
    out_col = row
    for i in range(nC):
        inputs[:, i, :, :out_col] = \
            inputs[:, i, :, int(step * i):int(step * i) + out_col]
    return inputs[:, :, :, :out_col]



class Illumination_Estimator(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):  #__init__部分是内部属性，而forward的输入才是外部输入
        super(Illumination_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w
        
        mean_c = img.mean(dim=1).unsqueeze(1)
        # stx()
        input = torch.cat([img,mean_c], dim=1)

        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map



class IG_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, illu_fea_trans):
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        illu_attn = illu_fea_trans # illu_fea: b,c,h,w -> b,h,w,c
        q, k, v, illu_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2)))
        v = v * illu_attn
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)


class IGAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, illu_fea):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out


class Denoiser(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4]):
        super(Denoiser, self).__init__()
        self.dim = dim
        self.level = level

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                IGAB(
                    dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)
            ]))
            dim_level *= 2

        # Bottleneck
        self.bottleneck = IGAB(
            dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2,
                                   kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                IGAB(
                    dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim,
                    heads=(dim_level // 2) // dim),
            ]))
            dim_level //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea):
        """
        x:          [b,c,h,w]         x是feature, 不是image
        illu_fea:   [b,c,h,w]
        return out: [b,c,h,w]
        """

        # Embedding
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        illu_fea_list = []
        for (IGAB, FeaDownSample, IlluFeaDownsample) in self.encoder_layers:
            fea = IGAB(fea,illu_fea)  # bchw
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            illu_fea = IlluFeaDownsample(illu_fea)

        # Bottleneck
        fea = self.bottleneck(fea,illu_fea)

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(
                torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.level-1-i]
            fea = LeWinBlcok(fea,illu_fea)

        # Mapping
        out = self.mapping(fea) + x

        return out


class RetinexFormer_Single_Stage(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, level=2, num_blocks=[1, 1, 1]):
        super(RetinexFormer_Single_Stage, self).__init__()
        self.estimator = Illumination_Estimator(n_feat)
        self.denoiser = Denoiser(in_dim=in_channels,out_dim=out_channels,dim=n_feat,level=level,num_blocks=num_blocks)  #### 将 Denoiser 改为 img2img
    
    def forward(self, img):
        # img:        b,c=3,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        illu_fea, illu_map = self.estimator(img)
        input_img = img * illu_map + img
        output_img = self.denoiser(input_img,illu_fea)

        return output_img


class RetinexFormer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, stage=3, num_blocks=[1,1,1]):
        super(RetinexFormer, self).__init__()
        # self.stage = stage

        # modules_body = [RetinexFormer_Single_Stage(in_channels=in_channels, out_channels=out_channels, n_feat=n_feat, level=2, num_blocks=num_blocks)
        #                 for _ in range(stage)]
        
        # self.body = nn.Sequential(*modules_body)
        self.body = Transformer(in_channels=in_channels, out_channels=out_channels, n_feat=n_feat)
    
    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        out = self.body(x)

        return out

## ====== MODIFY FROM DOWN HERE =============================================================================
import numbers
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from pdb import set_trace as stx
# import cv2

##########################################################################
## Utils

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

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
        return x / torch.sqrt(sigma+1e-5) * self.weight

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
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
    
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

##########################################################################
## Attention (Restormer-like)
class MHSA(nn.Module):
    def __init__(self, dim, num_heads, bias=True):
        super(MHSA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.proj_q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)    # Query projection (secondary)
        self.proj_kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)  # Key and Value projection (primary)
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape  # Get shape from the primary tensor
        
        # Calculate query from the secondary input
        q = self.proj_q(x) 
        # Calculate key and value from the primary input
        kv = self.proj_kv(x) 
        k, v = kv.chunk(2, dim=1)

        # Reshape for multi-head attention
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # Normalize q and k for better attention computation
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # Attention: compute dot-product attention
        attn = (q @ k.transpose(-2, -1))  # Scaled dot-product attention
        attn = attn.softmax(dim=-1)

        # Multiply attention by value
        out = attn @ v
        
        # Reshape back to original dimensions
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # Apply output projection
        out = self.proj_out(out)

        return out


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MHSA_Thermal(nn.Module):
    def __init__(self, dim, num_heads, bias=True):
        super(MHSA_Thermal, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Query-Key-Value projections
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)  # from RGB
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)  # from adapted Thermal
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)  # from adapted Thermal

        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x_rgb, x_thermal):
        """
        Cross-attention injection between RGB and Thermal
        Args:
            x_rgb: [B, C, H, W]  - main stream
            x_thermal: [B, 1, H, W] - auxiliary stream
        """
        B, C, H, W = x_rgb.shape


        # 2️⃣ Q from RGB, K/V from adapted Thermal
        q = self.q_proj(x_rgb)
        k = self.k_proj(x_thermal)
        v = self.v_proj(x_thermal)

        # 3️⃣ Multi-head reshape
        q = rearrange(q, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)
        k = rearrange(k, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)
        v = rearrange(v, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)

        # 4️⃣ Normalize
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        # 5️⃣ Compute attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # 6️⃣ Aggregate values
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h c (h1 w1) -> b (h c) h1 w1', h=self.num_heads, h1=H, w1=W)

        # 7️⃣ Project fused output
        out = self.out_proj(out)

        return out

    
import torch
from torch import nn
from einops import rearrange


class CrossModalMHSA(nn.Module):
    def __init__(self, dim, num_heads=8, bias=True):
        super(CrossModalMHSA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # === RGB projections (for Q and K only) ===
        self.proj_q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.proj_k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # === Value fusion ===
        # For Step 1 (spatial maxpool fusion)
        self.proj_spatial = nn.Conv2d(2, 2 * dim, kernel_size=1, bias=bias)

        # For Step 2 (GAP fusion)
        self.proj_global = nn.Conv1d(2, 2 * dim, kernel_size=1, bias=bias)

        # Output projection
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x_rgb, x_thermal, x_depth):
        B, C, H, W = x_rgb.shape

        # === STEP 1: Compute Q and K from RGB ===
        q = self.proj_q(x_rgb)
        k = self.proj_k(x_rgb)

        # === STEP 2: Compute V from thermal + depth ===
        # ----- Spatial MaxPool fusion -----
        t_pool = F.adaptive_max_pool2d(x_thermal, (H, W))  # [B, C, H, W]
        d_pool = F.adaptive_max_pool2d(x_depth, (H, W))    # [B, C, H, W]
        # Reduce across channels to get H×W×1 per modality
        t_map = t_pool.max(dim=1, keepdim=True)[0]          # [B,1,H,W]
        d_map = d_pool.max(dim=1, keepdim=True)[0]          # [B,1,H,W]
        spatial_cat = torch.cat([t_map, d_map], dim=1)      # [B,2,H,W]
        spatial_feat = self.proj_spatial(spatial_cat)       # [B,2C,H,W]
        spatial_feat = rearrange(spatial_feat, 'b c h w -> b (h w) c')  # [B,HW,2C]

        # ----- Global Average Pool fusion -----
        t_gap = F.adaptive_avg_pool2d(x_thermal, 1).view(B, C)  # [B,C]
        d_gap = F.adaptive_avg_pool2d(x_depth, 1).view(B, C)    # [B,C]
        global_cat = torch.stack([t_gap, d_gap], dim=1)         # [B,2,C]
        global_feat = self.proj_global(global_cat)              # [B,C,C] (2→C conv1d)

        # ----- Combine both steps -----
        # Expand global_feat to match spatial positions
        v = torch.matmul(spatial_feat, global_feat)  # [B,HW,1]
        v = v.permute(0, 2, 1)
        v = rearrange(v, 'b c (h w) -> b c h w', h=H, w=W)      # [B,C,H,W]


        # === Multi-head reshape ===
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # Normalize
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # === Attention ===
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)

        # === Output projection ===
        out = self.proj_out(out)
        return out







##########################################################################
import torch.nn.functional as F
# RetinexFormer FFN  
class FFN(nn.Module):
    def __init__(self, dim, mult=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )


    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x)


        return out



        
import os
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

import torch
import torchvision.transforms as transforms
from PIL import Image
##########################################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, in_channels, num_heads, dim_feedforward=None, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()
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


## Transformer Block
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
        # x_base = (x_rgb + x_thermal + x_depth) / 3.0
        # attn_out = self.attention(x_rgb, x_thermal, x_depth)
        # x = x_base + self.norm1(attn_out)
        # x = x + self.norm2(self.ffn(x))

        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet152, ResNet50_Weights, ResNet152_Weights


######################################################################
# Model
import torch
import torch.nn as nn

class Multeenhancer(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, n_feat=32, num_heads=2):
        super(Multeenhancer, self).__init__()

        # Input convs
        self.conv_in = nn.Conv2d(in_channels-1, n_feat, kernel_size=1, padding="same")
        self.thermal_convs = nn.ModuleList([
            nn.Conv2d(1, n_feat, kernel_size=3, padding=1),       # Level 1
            nn.Conv2d(1, n_feat*2, kernel_size=3, padding=1),     # Level 2
            nn.Conv2d(1, n_feat*4, kernel_size=3, padding=1),     # Bottleneck
            nn.Conv2d(1, n_feat*2, kernel_size=3, padding=1),     # Decoder Level 2
            nn.Conv2d(1, n_feat, kernel_size=3, padding=1),       # Decoder Level 1
        ])

        # ---------------- Encoder ----------------
        self.transformer_block1_1 = TransformerBlock(n_feat, num_heads)
        self.down1 = nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=2, padding=1)

        self.transformer_block2_1 = TransformerBlock(n_feat*2, num_heads)
        self.transformer_block2_2 = TransformerBlock(n_feat*2, num_heads)
        self.down2 = nn.Conv2d(n_feat*2, n_feat*4, kernel_size=3, stride=2, padding=1)

        # ---------------- Bottleneck ----------------
        self.bottleneck = TransformerBlock(n_feat*4, num_heads*2)

        # ---------------- Decoder ----------------
        self.up2 = nn.ConvTranspose2d(n_feat*4, n_feat*2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.channel_adjust2 = nn.Conv2d(n_feat*4, n_feat*2, kernel_size=1)
        self.transformer_block_up2_1 = TransformerBlock(n_feat*2, num_heads)

        self.up1 = nn.ConvTranspose2d(n_feat*2, n_feat, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.channel_adjust1 = nn.Conv2d(n_feat*2, n_feat, kernel_size=1)
        self.transformer_block_up1_1 = TransformerBlock(n_feat, num_heads)

        # ---------------- Output ----------------
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=1, padding="same")


    def forward(self, x):

        x_rgb = x[:, :3, :, :]        # [B, 3, H, W]
        x_thermal = x[:, 3:, :, :]    # [B, 1, H, W]
        # Adapt thermal for each level
        t1 = self.thermal_convs[0](x_thermal)
        t2 = self.thermal_convs[1](F.interpolate(x_thermal, scale_factor=0.5, mode='bilinear', align_corners=False))
        t3 = self.thermal_convs[2](F.interpolate(x_thermal, scale_factor=0.25, mode='bilinear', align_corners=False))
        t4 = self.thermal_convs[3](F.interpolate(x_thermal, scale_factor=0.5, mode='bilinear', align_corners=False))
        t5 = self.thermal_convs[4](x_thermal)

        # Input projection
        x = self.conv_in(x_rgb)

        # ---------------- Downward ----------------
        x1 = self.transformer_block1_1(x, t1)
        x1_down = self.down1(x1)

        x2 = self.transformer_block2_1(x1_down, t2)
        x2 = self.transformer_block2_2(x2, t2)
        x2_down = self.down2(x2)

        # ---------------- Bottleneck ----------------
        bn = self.bottleneck(x2_down, t3)

        # ---------------- Upward ----------------
        x2_up = self.up2(bn)
        x2_up = torch.cat([x2_up, x2], dim=1)
        x2_up = self.channel_adjust2(x2_up)
        x2_up = self.transformer_block_up2_1(x2_up, t4)
        x2_up = self.transformer_block_up2_2(x2_up, t4)

        x1_up = self.up1(x2_up)
        x1_up = torch.cat([x1_up, x1], dim=1)
        x1_up = self.channel_adjust1(x1_up)
        x1_up = self.transformer_block_up1_1(x1_up, t5)

        # Output
        out = self.conv_out(x1_up)
        return out

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling
    Captures multi-scale context through parallel dilated convolutions
    """
    def __init__(self, in_channels, out_channels=2048, atrous_rates=[6, 12, 18]):
        super().__init__()
        
        modules = []
        
        # 1x1 convolution
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Dilated convolutions
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        # Global average pooling
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        self.convs = nn.ModuleList(modules)
        
        # Project concatenated features
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        
        # Upsample global pooling to match spatial dimensions
        res[-1] = F.interpolate(res[-1], size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate all features
        res = torch.cat(res, dim=1)
        return self.project(res)


import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet152, resnet18, resnet101,  ResNet50_Weights, ResNet152_Weights, ResNet18_Weights, ResNet101_Weights

class FusionModule(nn.Module):
    def __init__(self, variant="resnet101", pretrained=True, freeze=False):
        super(FusionModule, self).__init__()
        
        if variant == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            resnet = resnet50(weights=weights)
        elif variant == "resnet152":
            weights = ResNet152_Weights.DEFAULT if pretrained else None
            resnet = resnet152(weights=weights)
        elif variant == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            resnet = resnet18(weights=weights)
        elif variant == "resnet101":
            weights = ResNet101_Weights.DEFAULT if pretrained else None
            resnet = resnet101(weights=weights)
        
        # --- Asymmetric Stems ---
        
        # 1. RGB Stem (pretrained)
        self.stem_r = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        
        # 2. Thermal Stem (new weights)
        # Use a new Conv2d for 1-channel input and a new BatchNorm2d
        self.conv1_t = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_t = nn.BatchNorm2d(64)
        self.stem_t = nn.Sequential(self.conv1_t, self.bn1_t, resnet.relu, resnet.maxpool)

        # 3. Depth Stem (new weights)
        # Use a new Conv2d for 1-channel input and a new BatchNorm2d
        self.conv1_d = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_d = nn.BatchNorm2d(64)
        self.stem_d = nn.Sequential(self.conv1_d, self.bn1_d, resnet.relu, resnet.maxpool)

        # --- Shared Layers ---
        self.layer1 = resnet.layer1  # 256 channels, H/4
        self.layer2 = resnet.layer2  # 512 channels, H/8
        self.layer3 = resnet.layer3  # 1024 channels, H/16
        self.layer4 = resnet.layer4  # 2048 channels, H/32
        
        if freeze:
            # Freeze all parameters, including new stems
            for p in self.parameters():
                p.requires_grad = False
            
            # If you only want to freeze the shared backbone, you would do:
            # for p in self.stem_r.parameters():
            #     p.requires_grad = False
            # for p in self.layer1.parameters():
            #     p.requires_grad = False
            # ... and so on for layer2, 3, 4
        
        # --- Adapters are no longer needed ---
        # self.adapt_conv_thermal = ... (REMOVED)
        # self.adapt_conv_depth = ... (REMOVED)
        
        # Cross-modal transformers
        self.cross_trans_1 = TransformerBlock_Fusion(256, num_heads=4)
        self.cross_trans_2 = TransformerBlock_Fusion(512, num_heads=4)
        self.cross_trans_3 = TransformerBlock_Fusion(1024, num_heads=8)
        self.cross_trans_4 = TransformerBlock_Fusion(2048, num_heads=16)
        
        # ---------------- Decoder ----------------
        self.up3 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.channel_adjust3 = nn.Conv2d(2048, 1024, kernel_size=1)
        self.transformer_block_up3_1 = TransformerBlock_Simple(1024, num_heads=8)

        self.up2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.channel_adjust2 = nn.Conv2d(1024, 512, kernel_size=1)
        self.transformer_block_up2_1 = TransformerBlock_Simple(512, num_heads=4)

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.channel_adjust1 = nn.Conv2d(512, 256, kernel_size=1)
        self.transformer_block_up1_1 = TransformerBlock_Simple(256, num_heads=4)

        # Auxiliary heads for deep supervision
        self.aux_head1 = nn.Conv2d(1024, 5, 1)
        self.aux_head2 = nn.Conv2d(512, 5, 1)

        self.aspp = ASPP(in_channels=2048, out_channels=2048, atrous_rates=[6,12,18])
    
    def forward(self, x_rgb, x_thermal, x_depth):
        # Store input size for final upsampling
        _, _, H, W = x_rgb.shape

        # --- Adapters are no longer needed ---
        # x_thermal = self.adapt_conv_thermal(x_thermal)
        # x_depth = self.adapt_conv_depth(x_depth)

        # ---------------- Encoder ----------------
        # Use the new separate stems
        f0_r = self.stem_r(x_rgb)      # H/4
        f0_t = self.stem_t(x_thermal)  # H/4
        f0_d = self.stem_d(x_depth)    # H/4
        
        # Pass features through shared layers
        f1_r = self.layer1(f0_r)      # H/4, 256
        f1_t = self.layer1(f0_t)
        f1_d = self.layer1(f0_d)
        f1_f = self.cross_trans_1(f1_r, f1_t, f1_d)

        f2_r = self.layer2(f1_r + f1_f)   # H/8, 512
        f2_t = self.layer2(f1_t + f1_f)
        f2_d = self.layer2(f1_d + f1_f)
        f2_f = self.cross_trans_2(f2_r, f2_t, f2_d)

        f3_r = self.layer3(f2_r + f2_f)   # H/16, 1024
        f3_t = self.layer3(f2_t + f2_f)
        f3_d = self.layer3(f2_d + f2_f)
        f3_f = self.cross_trans_3(f3_r, f3_t, f3_d)

        f4_r = self.layer4(f3_r + f3_f)   # H/32, 2048
        f4_t = self.layer4(f3_t + f3_f)
        f4_d = self.layer4(f3_d + f3_f)
        f4_f = self.cross_trans_4(f4_r, f4_t, f4_d)

        f4_f = self.aspp(f4_f)

        # ---------------- Decoder ----------------
        # Stage 3: upsample from bottleneck (2048→1024)
        x = self.up3(f4_f)
        if x.shape[-2:] != f3_f.shape[-2:]:
            x = F.interpolate(x, size=f3_f.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, f3_f], dim=1)           # concat skip
        x = self.channel_adjust3(x)               # reduce channels
        x = self.transformer_block_up3_1(x)
        stage3_features = x

        # Stage 2: upsample (1024→512)
        x = self.up2(x)
        if x.shape[-2:] != f2_f.shape[-2:]:
            x = F.interpolate(x, size=f2_f.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, f2_f], dim=1)
        x = self.channel_adjust2(x)
        x = self.transformer_block_up2_1(x)
        stage2_features = x

        # Stage 1: upsample (512→256)
        x = self.up1(x)
        if x.shape[-2:] != f1_f.shape[-2:]:
            x = F.interpolate(x, size=f1_f.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, f1_f], dim=1)
        x = self.channel_adjust1(x)
        x = self.transformer_block_up1_1(x)

        # Final upsample to input resolution (×4)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

        aux_features = {
            'stage3': self.aux_head1(stage3_features),
            'stage2': self.aux_head2(stage2_features)
        }

        return x, aux_features


# Main network
import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import os
import numpy as np
import cv2

import contextlib
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


class Multeenhancer_General(nn.Module):
    """
    Simplified network for RGB + Thermal (depth removed):
      Input: x [B, 4, H, W]  (RGB + Thermal)
      Steps:
        - Extract RGB and Thermal channels
        - Call fusion_module(rgb, thermal)
        - Optionally save RGB/Thermal inputs for inspection
    """

    def __init__(
        self,
        use_llie: bool = False,
        load_multeenhancer_ckpt: str = None,
        save_dir: str = "./saved_inputs",
    ):
        super(Multeenhancer_General, self).__init__()

        self.fusion_module = FusionModule()
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # --- Image index counter ---
        self.save_index = 0

        # --- Multeenhancer (optional LLIE module) ---
        self.use_llie = use_llie
        if self.use_llie:
            if multeenhancer is not None:
                self.multeenhancer = multeenhancer
            else:
                class _IdentityLLIE(nn.Module):
                    def forward(self, x):
                        return x[:, :3, :, :]
                self.multeenhancer = _IdentityLLIE()
                if load_multeenhancer_ckpt is not None:
                    print(f"[Multeenhancer_General] load_multeenhancer_ckpt provided but no model class supplied.")
            for p in self.multeenhancer.parameters():
                p.requires_grad = False

        # --- Depth Anything V2 (HuggingFace) ---
        self.depth_variant = "depth-anything/Depth-Anything-V2-Small-hf"
        self.freeze_depth = True
        self.run_depth_in_nograd = True

        self.depth_processor = AutoImageProcessor.from_pretrained(self.depth_variant)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(self.depth_variant)

        if self.freeze_depth:
            for p in self.depth_model.parameters():
                p.requires_grad = False


    def forward(self, x):
        """
        x: [B, 4, H, W] => [R, G, B, T]
        Returns: fusion_module output (segmentation logits)
        """
        assert x.dim() == 4 and x.size(1) >= 4, "Input must be [B, 4, H, W]"
        B, C, H, W = x.shape

        print(x.shape)

        device = x.device

        # --- Extract channels ---
        thermal = x[:, 3:4, :, :]  # [B, 1, H, W]
        
        if self.use_llie:
            with torch.no_grad():
                rgb = self.multeenhancer(x)  # [B, 3, H, W]
        else:
            rgb = x[:, :3, :, :]  # [B, 3, H, W]

        rgb = rgb.float().clamp(0.0, 1.0)
        thermal = thermal.float().clamp(0.0, 1.0)

        # --- Depth Estimation via HuggingFace ---
        # Processor expects list of images or tensors normalized to [0,1]
        # We'll process each batch element individually for safety
        depths = []
        context = torch.no_grad() if self.run_depth_in_nograd else contextlib.nullcontext()
        with context:
            for i in range(B):
                img = rgb[i].detach().cpu()  # [3,H,W]
                img_np = img.permute(1, 2, 0).numpy()  # HWC for HF processor
                inputs = self.depth_processor(images=img_np, return_tensors="pt", do_rescale=False).to(device)
                outputs = self.depth_model(**inputs)
                post = self.depth_processor.post_process_depth_estimation(
                    outputs, target_sizes=[(H, W)]
                )
                depth = post[0]["predicted_depth"]  # [H,W]
                depth = depth.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                depths.append(depth)

        depth_resized = torch.cat(depths, dim=0).to(device)  # [B,1,H,W]

        # --- Normalize depth per-sample to [0,1] ---
        depth_normed = []
        for i in range(depth_resized.size(0)):
            d = depth_resized[i:i+1]
            dmin = d.amin(dim=[1,2,3], keepdim=True)
            dmax = d.amax(dim=[1,2,3], keepdim=True)
            d = (d - dmin) / (dmax - dmin + 1e-8)
            depth_normed.append(d.clamp(0, 1))
        depth_resized = torch.cat(depth_normed, dim=0)

        # --- Call fusion module (RGB + Thermal only) ---
        return self.fusion_module(rgb, thermal, depth_resized)

    def save_tensor_as_image(self, tensor, save_dir, name_prefix):
        """Save the first element of a tensor as PNG.
        - RGB tensors: [B, 3, H, W] or [3, H, W] → saved as RGB image.
        - Thermal tensors: [B, 1, H, W] or [1, H, W] → saved as grayscale image.
        """
        import os
        import torch
        import numpy as np
        import cv2

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{name_prefix}.png")

        tensor = tensor.detach().cpu()

        # --- Keep only first element if batch dimension exists ---
        if tensor.dim() == 4:
            tensor = tensor[0]  # [C, H, W]

        # --- Handle single-channel (thermal) ---
        if tensor.size(0) == 1:
            array = tensor.squeeze(0).numpy()
            array = (array * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(save_path, array)  # grayscale

        # --- Handle RGB (3-channel) ---
        elif tensor.size(0) == 3:
            array = tensor.numpy()
            array = np.transpose(array, (1, 2, 0))  # [H, W, C]
            array = (array * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(save_path, array[:, :, ::-1])  # RGB→BGR for OpenCV

        else:
            raise ValueError(f"Unexpected tensor shape for image saving: {tensor.shape}")



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


from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights



class Multeenhancer_General_Seg(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, n_feat=32, freeze_backbone=True, pretrained=True):
        super(Multeenhancer_General_Seg, self).__init__()
        # Keep your fusion part unchanged
        self.backbone = Multeenhancer_General()

        # Segmentation head takes feature map and outputs logits per pixel
        self.head = SegmentationHead(in_channels=256, num_classes=5)


    def forward(self, x):
        # x: [B, 4, H, W]
        fused, aux_data = self.backbone(x)  # fusion output: [B, C_fuse, H, W]

        logits = self.head(fused)  # [B, num_classes, H/32, W/32]
        # Upsample to original resolution
        return logits, aux_data



