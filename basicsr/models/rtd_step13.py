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


class CrossModalMHSA_LLIE(nn.Module):
    def __init__(self, dim, num_heads=8, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # === Primary modality (RGB) projections ===
        self.proj_q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.proj_k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # === Secondary modality (thermal) ===
        # Step 1: spatial fusion
        self.proj_spatial = nn.Conv2d(1, dim, kernel_size=1, bias=bias)

        # Step 2: global fusion
        self.proj_global = nn.Conv1d(1, dim, kernel_size=1, bias=bias)

        # Output projection
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x_rgb, x_thermal):
        B, C, H, W = x_rgb.shape

        # === STEP 1: Q, K from RGB ===
        q = self.proj_q(x_rgb)
        k = self.proj_k(x_rgb)

        # === STEP 2: V from thermal ===
        # -------- Spatial fusion --------
        t_pool = F.adaptive_max_pool2d(x_thermal, (H, W))     # [B,C,H,W]
        t_map = t_pool.max(dim=1, keepdim=True)[0]             # [B,1,H,W]
        spatial_feat = self.proj_spatial(t_map)                # [B,C,H,W]
        spatial_feat = rearrange(spatial_feat, 'b c h w -> b (h w) c')   # [B,HW,C]

        # -------- Global fusion --------
        t_gap = F.adaptive_avg_pool2d(x_thermal, 1).view(B, C)  # [B,C]
        global_feat = self.proj_global(t_gap.unsqueeze(1))      # [B,C,C]

        # -------- Combine spatial + global --------
        v = torch.matmul(spatial_feat, global_feat)             # [B,HW,C]
        v = rearrange(v, 'b (h w) c -> b c h w', h=H, w=W)

        # === Multi-head reshape ===
        q = rearrange(q, 'b (h c) H W -> b h c (H W)', h=self.num_heads)
        k = rearrange(k, 'b (h c) H W -> b h c (H W)', h=self.num_heads)
        v = rearrange(v, 'b (h c) H W -> b h c (H W)', h=self.num_heads)

        # Normalize q,k
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # === Attention ===
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h c (H W) -> b (h c) H W', H=H, W=W)

        return self.proj_out(out)







##########################################################################
import torch.nn.functional as F
# RetinexFormer FFN  
class FFN(nn.Module):
    def __init__(self, dim, mult=2):
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
class TransformerBlock_LLIE(nn.Module):
    def __init__(self, in_channels, num_heads, dim_feedforward=None, LayerNorm_type='WithBias'):
        super(TransformerBlock_LLIE, self).__init__()
        if dim_feedforward is None:
            dim_feedforward = in_channels * 2

        self.cross_attention = CrossModalMHSA_LLIE(in_channels, num_heads)
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

class LLIE_Enhancer(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, n_feat=32, num_heads=2):
        super(LLIE_Enhancer, self).__init__()

        # Input convs
        self.conv_in = nn.Conv2d(3, n_feat, kernel_size=1, padding="same")
        self.thermal_convs = nn.ModuleList([
            nn.Conv2d(1, n_feat, kernel_size=3, padding=1),       # Level 1
            nn.Conv2d(1, n_feat*2, kernel_size=3, padding=1),     # Level 2
            nn.Conv2d(1, n_feat*4, kernel_size=3, padding=1),     # Bottleneck
            nn.Conv2d(1, n_feat*2, kernel_size=3, padding=1),     # Decoder Level 2
            nn.Conv2d(1, n_feat, kernel_size=3, padding=1),       # Decoder Level 1
        ])

        # ---------------- Encoder ----------------
        self.transformer_block1_1 = TransformerBlock_LLIE(n_feat, num_heads)
        self.down1 = nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=2, padding=1)

        self.transformer_block2_1 = TransformerBlock_LLIE(n_feat*2, num_heads)
        self.transformer_block2_2 = TransformerBlock_LLIE(n_feat*2, num_heads)
        self.down2 = nn.Conv2d(n_feat*2, n_feat*4, kernel_size=3, stride=2, padding=1)

        # ---------------- Bottleneck ----------------
        self.bottleneck = TransformerBlock_LLIE(n_feat*4, num_heads*2)

        # ---------------- Decoder ----------------
        self.up2 = nn.ConvTranspose2d(n_feat*4, n_feat*2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.channel_adjust2 = nn.Conv2d(n_feat*4, n_feat*2, kernel_size=1)
        self.transformer_block_up2_1 = TransformerBlock_LLIE(n_feat*2, num_heads)

        self.up1 = nn.ConvTranspose2d(n_feat*2, n_feat, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.channel_adjust1 = nn.Conv2d(n_feat*2, n_feat, kernel_size=1)
        self.transformer_block_up1_1 = TransformerBlock_LLIE(n_feat, num_heads)

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
        # x2_up = self.transformer_block_up2_2(x2_up, t4)

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
class GatedFusion(nn.Module):
    """Learnable gating for skip connections"""
    def __init__(self, in_ch):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x_skip, x_dec):
        gate = self.gate(x_skip)
        return x_skip * gate + x_dec


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet18, resnet50, resnet101, resnet152,
    ResNet18_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
)

# NOTE: TransformerBlock_Fusion, TransformerBlock_Simple and GatedFusion
# must be defined/imported in your project. This class uses them.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, resnet101, resnet152
from torchvision.models.resnet import ResNet18_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights

# Assuming TransformerBlock_Fusion and TransformerBlock_Simple are already defined

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, resnet101, resnet152
from torchvision.models.resnet import (
    ResNet18_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
)

# ---------------------------------------------------
# 🔹 Balanced Multi-Modal Encoding Block
# ---------------------------------------------------
class BalancedEncoder(nn.Module):
    """Aligns feature scales and semantics across modalities."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.proj(self.norm(x)))


# ---------------------------------------------------
# 🔹 Dynamic Modality Gating
# ---------------------------------------------------
class DynamicGating(nn.Module):
    """Learns reliability weights for each modality dynamically."""
    def __init__(self, channels):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 3, 1),  # 3 modalities -> 3 gating weights
            nn.Sigmoid()
        )

    def forward(self, fr, ft, fd):
        # Compute global reliability weights per modality
        f_sum = (fr + ft + fd) / 3
        gates = self.gate_mlp(f_sum)  # shape [B, 3, 1, 1]
        w_r, w_t, w_d = gates[:, 0:1], gates[:, 1:2], gates[:, 2:3]

        # Weighted fusion
        fused = (w_r * fr + w_t * ft + w_d * fd) / (w_r + w_t + w_d + 1e-6)
        return fused, (w_r, w_t, w_d)


# ---------------------------------------------------
# 🔹 Fusion Module
# ---------------------------------------------------
import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights

class FusionModule(nn.Module):
    def __init__(self, main_variant="resnet50", aux_variant="resnet18", freeze_main=False):
        super(FusionModule, self).__init__()

        # 1. Build Backbones
        # ----------------------------------------------------------------
        self.rgb_net = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.thm_net = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.dep_net = resnet18(weights=ResNet18_Weights.DEFAULT)

        # 2. Adapt Input Layers (Keep Pretrained Weights)
        # ----------------------------------------------------------------
        def adapt_first_layer(net):
            old_conv = net.conv1
            new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                new_conv.weight.data = old_conv.weight.data.sum(dim=1, keepdim=True)
            net.conv1 = new_conv
            return net

        self.thm_net = adapt_first_layer(self.thm_net)
        self.dep_net = adapt_first_layer(self.dep_net)

        # 3. Extract Layers (Backbone Definitions)
        # ----------------------------------------------------------------
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

        # 4. Projections (CORRECTION: Separate layers for T and D)
        # ----------------------------------------------------------------
        dims_main = [256, 512, 1024, 2048] # ResNet50
        dims_aux  = [64, 128, 256, 512]    # ResNet18

        # Thermal Projections
        self.proj_t1 = nn.Conv2d(dims_aux[0], dims_main[0], 1, bias=False)
        self.proj_t2 = nn.Conv2d(dims_aux[1], dims_main[1], 1, bias=False)
        self.proj_t3 = nn.Conv2d(dims_aux[2], dims_main[2], 1, bias=False)
        self.proj_t4 = nn.Conv2d(dims_aux[3], dims_main[3], 1, bias=False)

        # Depth Projections (Must be separate from Thermal!)
        self.proj_d1 = nn.Conv2d(dims_aux[0], dims_main[0], 1, bias=False)
        self.proj_d2 = nn.Conv2d(dims_aux[1], dims_main[1], 1, bias=False)
        self.proj_d3 = nn.Conv2d(dims_aux[2], dims_main[2], 1, bias=False)
        self.proj_d4 = nn.Conv2d(dims_aux[3], dims_main[3], 1, bias=False)

        # 5. Fusion Transformers
        # ----------------------------------------------------------------
        self.cross_trans_1 = TransformerBlock_Fusion(dims_main[0], num_heads=4)
        self.cross_trans_2 = TransformerBlock_Fusion(dims_main[1], num_heads=4)
        self.cross_trans_3 = TransformerBlock_Fusion(dims_main[2], num_heads=8)
        self.cross_trans_4 = TransformerBlock_Fusion(dims_main[3], num_heads=16)

        # 6. Decoder
        # ----------------------------------------------------------------
        self.up3 = nn.ConvTranspose2d(2048, 2048, 3, 2, 1, 1)
        self.channel_adjust3 = nn.Conv2d(2048 + 1024, 1024, 1)
        self.transformer_block_up3_1 = TransformerBlock_Simple(1024, num_heads=8)

        self.up2 = nn.ConvTranspose2d(1024, 1024, 3, 2, 1, 1)
        self.channel_adjust2 = nn.Conv2d(1024 + 512, 512, 1)
        self.transformer_block_up2_1 = TransformerBlock_Simple(512, num_heads=4)

        self.up1 = nn.ConvTranspose2d(512, 512, 3, 2, 1, 1)
        self.channel_adjust1 = nn.Conv2d(512 + 256, 256, 1)
        self.transformer_block_up1_1 = TransformerBlock_Simple(256, num_heads=4)

        self.aux_head1 = nn.Conv2d(1024, 5, 1)
        self.aux_head2 = nn.Conv2d(512, 5, 1)

        # 7. CRITICAL FIX FOR BATCH SIZE 2: Freeze Backbone BN
        # ----------------------------------------------------------------
        # We must lock the stats of the backbones. They cannot learn on BS=2.
        def freeze_bn(module):
            for m in module.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False
        
        # Apply to all backbone components
        for module in [self.stem_r, self.layer1_r, self.layer2_r, self.layer3_r, self.layer4_r,
                       self.stem_t, self.layer1_t, self.layer2_t, self.layer3_t, self.layer4_t,
                       self.stem_d, self.layer1_d, self.layer2_d, self.layer3_d, self.layer4_d]:
            freeze_bn(module)

    def forward(self, x_rgb, x_t, x_d):
        H, W = x_rgb.shape[2], x_rgb.shape[3]

        # --- Stems ---
        # We overwrite variables to save VRAM when possible
        f_r = self.stem_r(x_rgb)
        f_t = self.stem_t(x_t)
        f_d = self.stem_d(x_d)

        # --- Layer 1 ---
        f1_r = self.layer1_r(f_r)
        f1_t = self.layer1_t(f_t)
        f1_d = self.layer1_d(f_d)
        
        # Free stem memory immediately
        del f_r, f_t, f_d 

        # Project & Fuse
        # Note: We use separate projections proj_t1 and proj_d1
        f_f = self.cross_trans_1(f1_r, self.proj_t1(f1_t), self.proj_d1(f1_d))
        f1_r = f1_r + f_f
        
        # Save for Decoder skip connection
        skip1 = f_f 
        del f_f # Clean up temp

        # --- Layer 2 ---
        f2_r = self.layer2_r(f1_r)
        f2_t = self.layer2_t(f1_t)
        f2_d = self.layer2_d(f1_d)
        
        # Free Layer 1 Aux memory
        del f1_t, f1_d

        f_f = self.cross_trans_2(f2_r, self.proj_t2(f2_t), self.proj_d2(f2_d))
        f2_r = f2_r + f_f
        
        skip2 = f_f
        del f_f

        # --- Layer 3 ---
        f3_r = self.layer3_r(f2_r)
        f3_t = self.layer3_t(f2_t)
        f3_d = self.layer3_d(f2_d)
        
        # Free Layer 2 Aux memory
        del f2_t, f2_d

        f_f = self.cross_trans_3(f3_r, self.proj_t3(f3_t), self.proj_d3(f3_d))
        f3_r = f3_r + f_f
        
        skip3 = f_f
        del f_f

        # --- Layer 4 ---
        f4_r = self.layer4_r(f3_r)
        f4_t = self.layer4_t(f3_t)
        f4_d = self.layer4_d(f3_d)
        
        # Free Layer 3 Aux memory
        del f3_t, f3_d

        # Fusion at Bottleneck
        # Since this is the final semantic abstraction, we use the fused result directly
        f4_final = self.cross_trans_4(f4_r, self.proj_t4(f4_t), self.proj_d4(f4_d))
        
        # Free Layer 4 Aux memory
        del f4_r, f4_t, f4_d

        # ---------------- Decoder ----------------
        # 1. Up 3
        x = self.up3(f4_final)
        # Resize to match skip connection exactly (handles potential rounding errors in pooling)
        if x.shape[-2:] != skip3.shape[-2:]:
            x = F.interpolate(x, size=skip3.shape[-2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, skip3], dim=1)
        del skip3 # Clean up
        
        x = self.channel_adjust3(x)
        stage3 = self.transformer_block_up3_1(x)

        # 2. Up 2
        x = self.up2(stage3)
        if x.shape[-2:] != skip2.shape[-2:]:
            x = F.interpolate(x, size=skip2.shape[-2:], mode='bilinear', align_corners=False)
            
        x = torch.cat([x, skip2], dim=1)
        del skip2
        
        x = self.channel_adjust2(x)
        stage2 = self.transformer_block_up2_1(x)

        # 3. Up 1
        x = self.up1(stage2)
        if x.shape[-2:] != skip1.shape[-2:]:
            x = F.interpolate(x, size=skip1.shape[-2:], mode='bilinear', align_corners=False)
            
        x = torch.cat([x, skip1], dim=1)
        del skip1
        
        x = self.channel_adjust1(x)
        x = self.transformer_block_up1_1(x)

        # Final Resize
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

        aux = {"stage3": self.aux_head1(stage3), "stage2": self.aux_head2(stage2)}
        return x, aux






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
import albumentations as A



import torch
import torch.nn as nn
import torchvision.transforms as T
import kornia.augmentation as K  # pip install kornia

class Multeenhancer_General(nn.Module):
    def __init__(self, use_llie=False, llie_trainable=False, depth_variant="depth-anything/Depth-Anything-V2-Small-hf", **kwargs):
        super(Multeenhancer_General, self).__init__()

        self.fusion_module = FusionModule(**kwargs) # Pass variant here
        self.use_llie = use_llie
        
        # ... [LLIE loading logic same as before] ...

        # Depth Setup
        self.depth_processor = AutoImageProcessor.from_pretrained(depth_variant)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(depth_variant)
        self.depth_model.eval() # Always eval for depth anything
        for p in self.depth_model.parameters():
            p.requires_grad = False

        # --- FIX 1: GPU Augmentations (Differentiable) ---
        # Apply same geometric transform to all modalities, independent color to RGB
        self.aug_geo = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.2),
            K.RandomAffine(degrees=15, translate=(0.0625, 0.0625), scale=(0.8, 1.2), p=0.5),
            data_keys=["input", "input", "input"], # RGB, Thermal, Depth
            same_on_batch=False,
        )
        
        self.aug_color = K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8)

        # --- FIX 2: Normalization for ResNet Backbone ---
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x):
        # x: [B, 4, H, W]
        rgb = x[:, :3, :, :]
        thermal = x[:, 3:, :, :] # [B, 1, H, W]

        # 1. LLIE
        if self.use_llie:
            # Keep gradients if trainable
            if not any(p.requires_grad for p in self.llie_enhancer.parameters()):
                with torch.no_grad():
                    rgb = self.llie_enhancer(torch.cat([rgb, thermal], dim=1))
            else:
                rgb = self.llie_enhancer(torch.cat([rgb, thermal], dim=1))
        
        # Clamp to valid range for Depth model
        rgb = rgb.clamp(0, 1) 
        thermal = thermal.clamp(0, 1)

        # 2. Depth Estimation (Batched & Optimized)
        # Depth Anything expects normalized inputs usually, but processor handles it. 
        # Since we are inside model, manual resize/norm is faster than processor loop.
        with torch.no_grad():
            # Resize to closest multiple of 14 for ViT (Depth Anything requirement usually)
            h, w = rgb.shape[2], rgb.shape[3]
            # Note: Ideally, use the processor's transforms, but here is a simplified batched inference
            # Assuming input is roughly standard size. 
            # If strictly using processor:
            # inputs = self.depth_processor(images=rgb, return_tensors="pt") -> This puts on CPU.
            # Better to feed RGB tensor directly if model supports it, or standard normalization:
            
            # Simple approximation for Depth Anything input (ImageNet Norm)
            depth_input = (rgb - torch.tensor([0.485, 0.456, 0.406], device=rgb.device).view(1,3,1,1)) / \
                          torch.tensor([0.229, 0.224, 0.225], device=rgb.device).view(1,3,1,1)
            
            depth_out = self.depth_model(depth_input).predicted_depth # [B, H, W]
            depth = depth_out.unsqueeze(1) # [B, 1, H, W]

            
            # Resize back if depth model changed size
            if depth.shape[-2:] != (h, w):
                depth = F.interpolate(depth, size=(h, w), mode='bilinear', align_corners=False)

            # Min-Max Normalize Depth
            d_min = depth.amin(dim=(2, 3), keepdim=True)
            d_max = depth.amax(dim=(2, 3), keepdim=True)
            depth = (depth - d_min) / (d_max - d_min + 1e-8)

        # 3. Augmentation (GPU + Differentiable)
        # if self.training:
        #     # Geo transform (Shared)
        #     rgb, thermal, depth = self.aug_geo(rgb, thermal, depth)
        #     # Color transform (RGB only)
        #     rgb = self.aug_color(rgb)

        # 4. Normalize for ResNet Backbone
        # RGB needs ImageNet Norm. Thermal/Depth are 0-1, handled by custom Stems.
        rgb_norm = self.normalize(rgb)

        # 5. Fusion
        seg_out, aux = self.fusion_module(rgb_norm, thermal, depth)
        
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


class LLIE_SEG(nn.Module):
    """
    Experimental Model: ResNet-101 (stem + layer1 + layer2) -> Decoder -> SegHead
    
    Input: [B, 3, H, W] (RGB Image)
    Output: [B, num_classes, H, W] (Segmentation Logits)
    """
    def __init__(self, in_channels=5, out_channels=32, n_feat=32, num_classes=5,):
        super(LLIE_SEG, self).__init__()

        # --- 1. Load Backbone (ResNet-101) ---
        weights = ResNet101_Weights.DEFAULT if True else None
        resnet = resnet101(weights=weights)

        # --- 2. Encoder (stem, layer1, layer2) ---
        # Input: [B, 3, H, W]
        self.stem = nn.Sequential(
            resnet.conv1, 
            resnet.bn1, 
            resnet.relu, 
            resnet.maxpool
        )  # Output: [B, 64, H/4, W/4]
        
        self.layer1 = resnet.layer1  # Output: [B, 256, H/4, W/4]
        self.layer2 = resnet.layer2  # Output: [B, 512, H/8, W/8]

        # --- 3. Decoder (Upsampling) ---
        # We need to go from 512 channels at H/8 
        # to 32 channels at H (to match the SegHead's internal dim)
        self.decoder = nn.Sequential(
            # From [B, 512, H/8, W/8] -> [B, 64, H/4, W/4]
            # *** UPDATED in_channels from 128 to 512 ***
            nn.ConvTranspose2d(512, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # From [B, 64, H/4, W/4] -> [B, 32, H/2, W/2]
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # From [B, 32, H/2, W/2] -> [B, 32, H, W]
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # --- 4. Segmentation Head ---
        # Takes the upsampled 32-channel features
        self.head = SegmentationHead(in_channels=32, num_classes=5)

    def forward(self, x):
        """
        x: Input image, shape [B, 3, H, W]
        """
        
        x = x[:, :3, :, :]  # [B, 3, H, W]
        # --- Encoder ---
        f_stem = self.stem(x)
        f_l1 = self.layer1(f_stem)
        f_l2 = self.layer2(f_l1)
        
        # --- Decoder ---
        f_up = self.decoder(f_l2)
        
        # --- Head ---
        logits = self.head(f_up)
        
        return logits



class FocalLoss(nn.Module):
    """
    Implements Focal Loss, a SOTA alternative to Cross-Entropy for
    imbalanced datasets and hard-pixel mining.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', class_weights=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.register_buffer('class_weights', class_weights)

    def forward(self, pred, target):
        """
        pred: [B, C, H, W] logits
        target: [B, H, W] class indices
        """
        ce_loss = F.cross_entropy(pred, target, reduction='none', weight=self.class_weights)
        
        # Calculate the probability of the correct class
        pred_soft = F.softmax(pred, dim=1)
        p = torch.gather(pred_soft, 1, target.unsqueeze(1)).squeeze(1)
        
        # Calculate the focal loss modulating factor
        modulating_factor = (1 - p) ** self.gamma
        
        # Apply alpha weighting (optional, balances positive/negative classes)
        if self.alpha is not None:
            # You can also make alpha a tensor of [C] weights
            alpha_weight = target.float().new_tensor([self.alpha]).expand_as(target)
            alpha_weight = torch.where(target > 0, alpha_weight, 1.0 - alpha_weight)
            modulating_factor = alpha_weight * modulating_factor

        focal_loss = modulating_factor * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss



class SegmentationLoss(nn.Module):
    """
    Combined Focal + Dice loss with explicit deep supervision weights.
    - Replaces CE with Focal Loss for hard pixel mining.
    - Uses a batch-stable Dice Loss.
    - Applies specific weights to each auxiliary head.
    """
    def __init__(self, 
                 weight_focal=2.0, 
                 weight_dice=1.5, 
                 smooth=1.0,
                 class_weights="/home/ancuti/expendables/Multeenhancer/data/PST900_RGBT_Dataset/weights.ini", 
                 num_classes=5, 
                 focal_gamma=2.5,
                 focal_alpha=0.25,
                 aux_weights={'stage1': 0.7, 'stage2': 0.3}):
        
        super(SegmentationLoss, self).__init__()
        self.weight_focal = weight_focal
        self.weight_dice = weight_dice
        self.smooth = smooth
        self.num_classes = num_classes
        self.aux_weights = aux_weights
        
        # Load and register class weights
        if isinstance(class_weights, str) and os.path.isfile(class_weights):
            class_weights = self._load_class_weights_from_ini(class_weights)
            print(f"[ImprovedLoss] Loaded class weights: {class_weights}")
        
        if isinstance(class_weights, (np.ndarray, list)):
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
            
        # 1. Use Focal Loss instead of CE
        self.focal = FocalLoss(
            alpha=focal_alpha, 
            gamma=focal_gamma, 
            class_weights=self.class_weights
        )

    def _load_class_weights_from_ini(self, weight_file):
        # (Your _load_class_weights_from_ini function is fine, keep it as-is)
        import configparser
        config = configparser.ConfigParser()
        config.read(weight_file)
        weights = np.zeros(self.num_classes)
        try:
            weights[0] = float(config['ClassWeights']['background'])
            weights[1] = float(config['ClassWeights']['fire_extinguisher'])
            weights[2] = float(config['ClassWeights']['backpack'])
            weights[3] = float(config['ClassWeights']['drill'])
            weights[4] = float(config['ClassWeights']['rescue_randy'])
        except KeyError:
            raise ValueError(f"Invalid or missing entries in {weight_file}")
        return weights
    
    def dice_loss(self, pred, target):
        """
        2. More Stable Dice Loss
        Calculates intersection and union over the whole batch for each class.
        """
        pred_soft = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        
        # Sum over (Batch, H, W) dims
        intersection = (pred_soft * target_onehot).sum(dim=(0, 2, 3))
        union = pred_soft.sum(dim=(0, 2, 3)) + target_onehot.sum(dim=(0, 2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Average the loss over all classes
        return (1 - dice).mean()

    def _compute_loss(self, pred, target):
        """Computes the combined loss for a single prediction."""
        loss_focal = self.focal(pred, target)
        loss_dice = self.dice_loss(pred, target)
        return self.weight_focal * loss_focal + self.weight_dice * loss_dice

    def forward(self, prediction, target):
        """
        prediction: Tuple (main_output, aux_outputs_dict)
        target: [B, H, W] ground truth
        """
        main_pred = prediction[0]
        aux_outputs = prediction[1]

        # Ensure target shape
        if target.ndim == 4:
            target = target.squeeze(1)
        if target.dtype != torch.long:
            target = target.long()
        
        # Main prediction loss
        main_loss = self._compute_loss(main_pred, target)
        
        # 3. Explicitly Weighted Deep Supervision
        if aux_outputs is not None and len(aux_outputs) > 0:
            aux_loss = 0
            for aux_name, aux_pred in aux_outputs.items():
                if aux_name in self.aux_weights:
                    # Resize target to match aux prediction
                    # This is more efficient than upsampling the prediction
                    target_aux = F.interpolate(
                        target.unsqueeze(1).float(),
                        size=aux_pred.shape[2:],
                        mode='nearest'
                    ).squeeze(1).long()
                    
                    aux_loss += self.aux_weights[aux_name] * self._compute_loss(aux_pred, target_aux)
            
            total_loss = main_loss + aux_loss
        else:
            total_loss = main_loss
        
        return total_loss


# general settings
name: Enhancement_RetinexFormer_LOL_v1
model_type: ImageCleanModel
scale: 1
num_gpu: 1  # set num_gpu: 0 for cpu mode
manual_seed: 100

# dataset and data loader settings
datasets:
  train:
    name: TrainSet
    type: Dataset_PairedImage
    dataroot_gt: data/PST900_RGBT_Dataset/train/labels/
    dataroot_lq: data/PST900_RGBT_Dataset/train/rgb/
    lq_second: data/PST900_RGBT_Dataset/train/thermal/
    geometric_augs: true
    train_size: [1280, 720]
    random_crop: [512, 512]

    filename_tmpl: '{}'
    io_backend:
      type: disk

    # data loader
    use_shuffle: true
    num_worker_per_gpu: 8
    batch_size_per_gpu: 4

    ### -------------Progressive training--------------------------
    # mini_batch_sizes: [8,8,4,2,1]             # Batch size per gpu   
    # iters: [46000,32000,24000,18000,18000,12000]
    # gt_size: 384   # Max patch size for progressive training
    # gt_sizes: [192,256,320,384,416]  # Patch sizes for progressive training.
    # ### ------------------------------------------------------------

    ### ------- Training on single fixed-patch size 128x128---------
    mini_batch_sizes: [4]   
    iters: [150000]
    gt_size: 256
    gt_sizes: [256]
    ### ------------------------------------------------------------

    dataset_enlarge_ratio: 1
    prefetch_mode: ~

  val:
    name: ValSet
    type: Dataset_PairedImage
    dataroot_gt: data/PST900_RGBT_Dataset/test/labels/
    dataroot_lq: data/PST900_RGBT_Dataset/test/rgb/
    lq_second: data/PST900_RGBT_Dataset/test/thermal/
    io_backend:
      type: disk

# network structures
network_g:
  type: Multeenhancer_General_Seg
  in_channels: 4
  out_channels: 1
  n_feat: 32


# path
path:
  pretrain_network_g: ~
  strict_load_g: true
  resume_state: ~

# training settings
train:
  total_iter: 40000
  warmup_iter: 1500 # no warm up
  use_grad_clip: true
  grad_clip_norm: 0.1
  

  # Split 300k iterations into two cycles. 
  # 1st cycle: fixed 3e-4 LR for 92k iters. 
  # 2nd cycle: cosine annealing (3e-4 to 1e-6) for 208k iters.
  scheduler:
    type: TrueCosineAnnealingLR
    T_max: 10000
    eta_min: !!float 1e-7
  
  mixing_augs:
    mixup: false
    use_identity: true


  optim_g:
    type: AdamW
    lr: !!float 3e-4
    weight_decay: !!float 0.05
    betas: [0.9, 0.999]
  
  
  # losses
  pixel_opt:
    type: SegmentationLoss


# validation settings
val:
  window_size: 4
  val_freq: !!float 500
  save_img: false
  rgb2bgr: false
  use_image: false
  max_minibatch: 8

  metrics:
    # psnr: # metric name, can be arbitrary
    #   type: calculate_psnr
    #   crop_border: 0
    #   test_y_channel: false
    absrel_mae:
        type: calculate_miou
# logging settings
logger:
  print_freq: 500
  save_checkpoint_freq: !!float 1e4
  use_tb_logger: true
  wandb:
    project: low_light
    resume_id: ~

# dist training settings
dist_params:
  backend: nccl
  port: 29500
