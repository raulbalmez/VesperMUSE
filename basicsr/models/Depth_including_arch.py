class FusionModule(nn.Module):
    def __init__(self, variant="resnet152", pretrained=True, freeze=False):
        super().__init__()

        # === Shared ResNet encoder (3 levels) ===
        if variant == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            resnet = resnet50(weights=weights)
        elif variant == "resnet152":
            weights = ResNet152_Weights.DEFAULT if pretrained else None
            resnet = resnet152(weights=weights)
        elif variant == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            resnet = resnet18(weights=weights)
        else:
            raise ValueError("variant must be 'resnet50' or 'resnet152'")

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.layer1 = resnet.layer1   # output 256
        self.layer2 = resnet.layer2   # output 512

        # Optional freeze
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

        # === Modality adaptors ===
        self.adapt_conv_thermal = nn.Conv2d(1, 3, kernel_size=1)
        self.adapt_conv_depth = nn.Conv2d(1, 3, kernel_size=1)

        # === Cross-modal transformers ===
        self.cross_trans_1 = TransformerBlock_Fusion(64, num_heads=2)
        self.cross_trans_2 = TransformerBlock_Fusion(256, num_heads=2)
        self.cross_trans_3 = TransformerBlock_Fusion(512, num_heads=4)

        # === Decoder (3 levels, convention consistent) ===
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.channel_adjust2 = nn.Conv2d(256 + 256, 256, kernel_size=1)
        self.trans_up2_1 = TransformerBlock_Simple(256, num_heads=2)

        self.up1 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.channel_adjust1 = nn.Conv2d(64 + 64, 64, kernel_size=1)
        self.trans_up1_1 = TransformerBlock_Simple(64, num_heads=2)

        self.up_sample = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)


    def forward(self, x_rgb, x_thermal, x_depth):
        # === Adapt inputs ===
        x_thermal = self.adapt_conv_thermal(x_thermal)
        x_depth = self.adapt_conv_depth(x_depth)

        # === Encoder ===
        f1_r = self.stem(x_rgb)
        f1_t = self.stem(x_thermal)
        f1_d = self.stem(x_depth)
        f1_f = self.cross_trans_1(f1_r, f1_t, f1_d)
        f1_r = f1_r + f1_f

        f2_r = self.layer1(f1_r)
        f2_t = self.layer1(f1_t)
        f2_d = self.layer1(f1_d)
        f2_f = self.cross_trans_2(f2_r, f2_t, f2_d)
        f2_r = f2_r + f2_f

        f3_r = self.layer2(f2_r)
        f3_t = self.layer2(f2_t)
        f3_d = self.layer2(f2_d)
        f3_f = self.cross_trans_3(f3_r, f3_t, f3_d)
        f3_r = f3_r + f3_f

        # === Decoder ===
        x = self.up2(f3_r)
        x = self.channel_adjust2(torch.cat([x, f2_r], dim=1))
        x = self.trans_up2_1(x)

        x = self.up1(x)
        x = self.channel_adjust1(torch.cat([x, f1_r], dim=1))
        x = self.trans_up1_1(x)

        out = self.up_sample(x)
        return out


    

# Segmentation former
import torchvision.models.segmentation as segmentation
from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights
# from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation



import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

import contextlib
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


class Multeenhancer_General(nn.Module):
    """
    Full network:
      Input: x [B, 4, H, W]  (RGB + Thermal)
      Steps:
        - Optionally run Multeenhancer LLIE (if use_llie True)
        - Run RGB through Depth Anything V2 (HuggingFace)
        - Resize depth to RGB size and normalize
        - Extract thermal channel and resize
        - Call fusion_module(thermal, rgb, depth)
    """

    def __init__(
        self,
        use_llie: bool = False,
        load_multeenhancer_ckpt: str = None,
        depth_variant: str = "depth-anything/Depth-Anything-V2-Small-hf",
        freeze_depth: bool = True,
        run_depth_in_nograd: bool = True,
    ):
        super(Multeenhancer_General, self).__init__()

        # --- Fusion Module placeholder (replace with actual) ---
        self.fusion_module = FusionModule()

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
                    print(f"[Multeenhancer_Depth] load_multeenhancer_ckpt provided but no model class supplied.")
            for p in self.multeenhancer.parameters():
                p.requires_grad = False

        # --- Depth Anything V2 (HuggingFace) ---
        self.depth_variant = depth_variant
        self.freeze_depth = freeze_depth
        self.run_depth_in_nograd = run_depth_in_nograd

        self.depth_processor = AutoImageProcessor.from_pretrained(depth_variant)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(depth_variant)

        if self.freeze_depth:
            for p in self.depth_model.parameters():
                p.requires_grad = False

        # --- Optional small projections (can be overridden externally) ---

    def forward(self, x):
        """
        x: [B,4,H,W] => [R,G,B,T]
        Returns: fusion_module(thermal, rgb, depth)
        """
        assert x.dim() == 4 and x.size(1) >= 4, "Input must be [B,4,H,W]"
        device = x.device
        B, C, H, W = x.shape

        # --- Thermal channel ---
        thermal = x[:, 3:4, :, :]  # [B,1,H,W]

        # --- RGB (optionally enhanced) ---
        if self.use_llie:
            with torch.no_grad() if self.run_depth_in_nograd else contextlib.nullcontext():
                rgb = self.multeenhancer(x)
        else:
            rgb = x[:, :3, :, :]
    

        rgb = rgb.float().clamp(0.0, 1.0)

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

        # --- Ensure consistent sizes and types ---
        if thermal.shape[-2:] != (H, W):
            thermal = F.interpolate(thermal, size=(H, W), mode='bilinear', align_corners=False)
        thermal = thermal.float()
        depth_resized = depth_resized.float()
        rgb = rgb.float()

        # --- Call fusion module ---
        return self.fusion_module(rgb, thermal, depth_resized)


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SegmentationHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            LayerNorm(dim=in_channels, LayerNorm_type="WithBias"),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            LayerNorm(dim=in_channels // 2, LayerNorm_type="WithBias"),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, num_classes, 1)
        )

    def forward(self, x):
        return self.conv(x)