import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
import torchvision.models as models
from pytorch_msssim import ms_ssim

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss   #把 l1_loss 作为 weighted_loss 的输入
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss   #把 mse_loss 作为 weighted_loss 的输入
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=True).features[:16]  # Until block3_conv3
        self.loss_model = vgg.to(device).eval()
        for param in self.loss_model.parameters():
            param.requires_grad = False

    def forward(self, y_true, y_pred):
        y_true, y_pred = y_true.to(next(self.loss_model.parameters()).device), y_pred.to(next(self.loss_model.parameters()).device)
        return F.mse_loss(self.loss_model(y_true), self.loss_model(y_pred))

def multiscale_ssim_loss(y_true, y_pred, max_val=1.0, power_factors=[0.5, 0.5]):
    return 1.0 - ms_ssim(y_true, y_pred, data_range=max_val, size_average=True)

class HybridLoss(nn.Module):
    def __init__(self, device=None, alpha1=1.0, alpha2=0.01, alpha3=0.2, alpha4 = 0.3):
        super(HybridLoss, self).__init__()
        if alpha1 != 0:
            self.l1_loss = MSELoss()
        
        if alpha2 != 0:
            self.perceptual_loss_model = None

        if alpha3 != 0:
            self.ms_ssim_loss = multiscale_ssim_loss
        

        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3

    def forward(self, y_true, y_pred):
        total_loss = 0

        if self.alpha1 != 0:
            l1_l = self.l1_loss(y_true, y_pred)
            total_loss += self.alpha1 * l1_l
        
        if self.alpha2 != 0:
            if self.perceptual_loss_model is None:
                self.perceptual_loss_model = VGGPerceptualLoss(y_true.device)
            perc_l = self.perceptual_loss_model(y_true, y_pred)
            total_loss += self.alpha2*perc_l 
        
        if self.alpha3 != 0:
            ms_ssim_l = self.ms_ssim_loss(y_true, y_pred)
            total_loss += self.alpha3*ms_ssim_l

        return torch.mean(total_loss)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class HybridLoss_SEG(nn.Module):
    def __init__(self, segmentation_feature_extractor=None, device=None, 
                 alpha1=1.0, alpha2=0.01, alpha3=0.2, alpha4=0.3):
        """
        Args:
            segmentation_feature_extractor: A pretrained segmentation encoder (e.g., ResNet backbone).
                                            Must return feature maps or logits.
        """
        super(HybridLoss_SEG, self).__init__()
        
        # 1. Pixel-wise Loss (MSE/L1) - For color correctness
        if alpha1 != 0:
            self.l1_loss = nn.L1Loss() # L1 is often sharper than MSE for image restoration
        
        # 2. VGG Perceptual Loss - For general visual quality
        if alpha2 != 0:
            self.perceptual_loss_model = None # Lazy loading handled in forward

        # 3. SSIM Loss - For structural integrity
        if alpha3 != 0:
            self.ms_ssim_loss = multiscale_ssim_loss
        
        # 4. Semantic Consistency Loss - FOR MAXIMIZING SEGMENTATION mIoU
        if alpha4 != 0:
            if segmentation_feature_extractor is None:
                # Fallback: Use a standard ResNet50 if no specific segmenter provided
                resnet = models.resnet50(pretrained=True)
                # We typically want the features before the final pooling
                self.seg_model = nn.Sequential(*list(resnet.children())[:-2]).to(device)
            else:
                self.seg_model = segmentation_feature_extractor.to(device)
            
            # Freeze the segmentation model (We don't want to train it, just use it as a ruler)
            self.seg_model.eval()
            for param in self.seg_model.parameters():
                param.requires_grad = False

        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.alpha4 = alpha4
        self.device = device

    def forward(self, y_true, y_pred):
        total_loss = 0
        loss_dict = {}

        # --- 1. Reconstruction Loss ---
        if self.alpha1 != 0:
            l1_l = self.l1_loss(y_true, y_pred)
            total_loss += self.alpha1 * l1_l
            loss_dict['l1'] = l1_l.item()
        
        # --- 2. Perceptual Loss (VGG) ---
        if self.alpha2 != 0:
            if self.perceptual_loss_model is None:
                # Ensure it's on the correct device
                self.perceptual_loss_model = VGGPerceptualLoss(y_true.device)
            perc_l = self.perceptual_loss_model(y_true, y_pred)
            total_loss += self.alpha2 * perc_l 
            loss_dict['perc'] = perc_l.item()
        
        # --- 3. SSIM Loss ---
        if self.alpha3 != 0:
            ms_ssim_l = self.ms_ssim_loss(y_true, y_pred)
            total_loss += self.alpha3 * ms_ssim_l
            loss_dict['ssim'] = ms_ssim_l.item()

        # --- 4. Semantic Consistency Loss (The Key Fix) ---
        if self.alpha4 != 0:
            # Normalize inputs for the segmentation model if necessary (usually ImageNet stats)
            # Assuming y_true and y_pred are [0, 1]
            # You might need to add standard normalization here: transform(y)
            
            with torch.no_grad():
                feat_true = self.seg_model(y_true)
            
            # We want gradients to flow back to the enhancer via y_pred
            feat_pred = self.seg_model(y_pred)
            
            # Compare the high-level semantic features
            # L1 Loss on features forces the "content" to be the same
            sem_l = F.l1_loss(feat_pred, feat_true)
            
            # Alternatively: Cosine Similarity Loss can sometimes work better for features
            # sem_l = 1 - F.cosine_similarity(feat_pred, feat_true).mean()

            total_loss += self.alpha4 * sem_l
            loss_dict['semantic'] = sem_l.item()

        return total_loss



import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class AbsRelMAELoss(nn.Module):
    def __init__(self, alpha1=1.0, alpha2=0.01, eps=1e-6, use_absrel=True):
        """
        Args:
            alpha1: Weight for depth loss
            alpha2: Weight for smoothness loss
            eps: Small constant for numerical stability
            use_absrel: If True, uses absolute relative error; if False, uses MAE
        """
        super(AbsRelMAELoss, self).__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.eps = eps
        self.use_absrel = use_absrel
    
    def charbonnier_loss(self, diff):
        """Charbonnier loss (smooth L1)"""
        return torch.sqrt(diff ** 2 + self.eps ** 2)
    
    def gradient(self, x):
        """Compute image gradients"""
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx, dy
    
    def forward(self, pred, target, image=None):
        # Clamp to valid range
        pred = torch.clamp(pred, self.eps, 1.0)
        target = torch.clamp(target, self.eps, 1.0)
        
        # Create validity mask
        mask = (target > self.eps) & (target <= 1.0) & ~torch.isnan(pred) & ~torch.isnan(target)
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # === Depth Loss ===
        pred_masked = pred[mask]
        target_masked = target[mask]
        
        if self.use_absrel:
            # Absolute Relative Error: |pred - target| / target
            abs_rel = torch.abs(pred_masked - target_masked) / (target_masked + self.eps)
            loss_depth = torch.mean(abs_rel)
        else:
            # MAE with Charbonnier
            diff = pred_masked - target_masked
            loss_depth = torch.mean(self.charbonnier_loss(diff))
        
        # === Smoothness Loss ===
        # Apply mask to gradients (only compute smoothness on valid regions)
        # Erode mask slightly for gradient computation
        mask_x = mask[:, :, :, :-1] & mask[:, :, :, 1:]
        mask_y = mask[:, :, :-1, :] & mask[:, :, 1:, :]
        
        dx_pred, dy_pred = self.gradient(pred)
        
        if image is not None:
            # Edge-aware smoothness (penalize less at image edges)
            dx_img, dy_img = self.gradient(image)
            weight_x = torch.exp(-torch.mean(torch.abs(dx_img), 1, keepdim=True))
            weight_y = torch.exp(-torch.mean(torch.abs(dy_img), 1, keepdim=True))
            
            # Apply mask to smoothness
            if mask_x.sum() > 0:
                loss_smooth_x = torch.mean(torch.abs(dx_pred[mask_x]) * weight_x[mask_x])
            else:
                loss_smooth_x = torch.tensor(0.0, device=pred.device)
            
            if mask_y.sum() > 0:
                loss_smooth_y = torch.mean(torch.abs(dy_pred[mask_y]) * weight_y[mask_y])
            else:
                loss_smooth_y = torch.tensor(0.0, device=pred.device)
            
            loss_smooth = loss_smooth_x + loss_smooth_y
        else:
            # Simple smoothness without image guidance
            if mask_x.sum() > 0:
                loss_smooth_x = torch.mean(torch.abs(dx_pred[mask_x]))
            else:
                loss_smooth_x = torch.tensor(0.0, device=pred.device)
            
            if mask_y.sum() > 0:
                loss_smooth_y = torch.mean(torch.abs(dy_pred[mask_y]))
            else:
                loss_smooth_y = torch.tensor(0.0, device=pred.device)
            
            loss_smooth = loss_smooth_x + loss_smooth_y
        
        # === Total Loss ===
        loss = self.alpha1 * loss_depth + self.alpha2 * loss_smooth
        
        return loss


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import configparser
import numpy as np
import os


class SegmentationLoss(nn.Module):
    """
    LASNet-style CE + Dice loss with automatic weight loading.
    """
    def __init__(self, weight_ce=1.0, weight_dice=0.5, smooth=1e-5,
                 class_weights="/home/ancuti/expendables/Multeenhancer/data/PST900_RGBT_Dataset/weights.ini", 
                 num_classes=5):
        super(SegmentationLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.smooth = smooth
        self.num_classes = num_classes
        
        # --- Load weights automatically if a path is provided ---
        if isinstance(class_weights, str) and os.path.isfile(class_weights):
            class_weights = self._load_class_weights_from_ini(class_weights)
            print(f"[SegmentationLoss] Loaded class weights: {class_weights}")
        
        # --- Convert to tensor and register as buffer ---
        if isinstance(class_weights, (np.ndarray, list)):
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
            # Option: Don't normalize, or normalize by min
            # class_weights = class_weights / class_weights.min()
        else:
            class_weights = None
        
        # Register as buffer so it moves with model
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
            self.ce = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.ce = nn.CrossEntropyLoss()
    
    def _load_class_weights_from_ini(self, weight_file):
        """Reads .ini file and returns numpy array of class weights."""
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
        Compute Dice loss over all classes.
        pred: [B, C, H, W] logits
        target: [B, H, W] class indices
        """
        pred_soft = F.softmax(pred, dim=1)  # [B, C, H, W]
        
        # One-hot encode target
        target_onehot = torch.zeros_like(pred_soft)
        target_onehot.scatter_(1, target.unsqueeze(1), 1)  # [B, C, H, W]
        
        # Calculate intersection and union per class
        intersection = (pred_soft * target_onehot).sum(dim=(2, 3))  # [B, C]
        union = pred_soft.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))  # [B, C]
        
        # Dice coefficient per class
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # [B, C]
        
        # Average over all classes and batch
        return 1 - dice.mean()
    
    def forward(self, prediction, target):
        """
        prediction: [B, C, H, W] logits
        target: [B, H, W] or [B, 1, H, W] class indices
        """

        # Ensure target is [B, H, W]
        if target.ndim == 4:
            target = target.squeeze(1)
        
        # Ensure target is long type
        if target.dtype != torch.long:
            target = target.long()
        
        # Compute losses
        loss_ce = self.ce(prediction, target)
        loss_dice = self.dice_loss(prediction, target)
        
        total_loss = self.weight_ce * loss_ce + self.weight_dice * loss_dice
        return total_loss


# SEG PST900 + MFNet
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
            # weights[0] = float(config['ClassWeights']['background'])
            # weights[1] = float(config['ClassWeights']['car'])
            # weights[2] = float(config['ClassWeights']['person'])
            # weights[3] = float(config['ClassWeights']['bike'])
            # weights[4] = float(config['ClassWeights']['curve'])
            # weights[5] = float(config['ClassWeights']['car stop'])
            # weights[6] = float(config['ClassWeights']['guardrail'])
            # weights[7] = float(config['ClassWeights']['color cone'])
            # weights[8] = float(config['ClassWeights']['bump'])
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
        # main_pred = prediction[0]
        # aux_outputs = prediction[1]

        main_pred = prediction
        aux_outputs = None

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

#SEG MFNEt + PST900

#SEG CART

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import os
# import configparser

# class FocalLoss(nn.Module):
#     """
#     Implements Focal Loss, a SOTA alternative to Cross-Entropy for
#     imbalanced datasets and hard-pixel mining.
#     """
#     def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', class_weights=None, ignore_index=255):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = reduction
#         self.ignore_index = ignore_index
#         self.register_buffer('class_weights', class_weights)

#     def forward(self, pred, target):
#         """
#         pred: [B, C, H, W] logits
#         target: [B, H, W] class indices
#         """
#         # 1. Calculate CE Loss (handles ignore_index internally for the loss value)
#         print("loss checking", pred.shape, target.shape)
#         ce_loss = F.cross_entropy(
#             pred, 
#             target, 
#             reduction='none', 
#             weight=self.class_weights, 
#             ignore_index=self.ignore_index
#         )
        
#         # 2. Calculate the probability of the correct class
#         pred_soft = F.softmax(pred, dim=1)
        
#         # Safe gather: mask out ignore_index to prevent out-of-bounds access
#         # We replace 255 with 0 just for the gather step; the loss will be masked later anyway
#         valid_mask = target != self.ignore_index
#         target_safe = target.clone()
#         target_safe[~valid_mask] = 0  
        
#         # Gather the probability of the true class
#         p = torch.gather(pred_soft, 1, target_safe.unsqueeze(1)).squeeze(1)
        
#         # 3. Calculate the focal loss modulating factor
#         modulating_factor = (1 - p) ** self.gamma
        
#         # 4. Apply alpha weighting (optional, balances positive/negative classes)
#         if self.alpha is not None:
#             alpha_weight = target.float().new_tensor([self.alpha]).expand_as(target)
#             alpha_weight = torch.where(target > 0, alpha_weight, 1.0 - alpha_weight)
#             modulating_factor = alpha_weight * modulating_factor

#         focal_loss = modulating_factor * ce_loss
        
#         # Explicitly zero out loss for ignored pixels (redundant if ce_loss is 0, but safe)
#         focal_loss = torch.where(valid_mask, focal_loss, torch.zeros_like(focal_loss))
        
#         if self.reduction == 'mean':
#             # normalize by number of valid pixels, not total pixels
#             return focal_loss.sum() / (valid_mask.sum() + 1e-6)
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         else:
#             return focal_loss


# class SegmentationLoss(nn.Module):
#     """
#     Combined Focal + Dice loss with explicit deep supervision weights.
#     - Replaces CE with Focal Loss for hard pixel mining.
#     - Uses a batch-stable Dice Loss.
#     - Applies specific weights to each auxiliary head.
#     - Automatically handles CART label mapping (shift -2).
#     """
#     def __init__(self, 
#                  weight_focal=2.0, 
#                  weight_dice=1.5, 
#                  smooth=1.0,
#                  class_weights="/home/ancuti/expendables/Multeenhancer/data/CART/cart_weights.ini", 
#                  num_classes=10, 
#                  focal_gamma=2.5,
#                  focal_alpha=0.25,
#                  ignore_index=255,
#                  aux_weights={'stage1': 0.7, 'stage2': 0.3}):
        
#         super(SegmentationLoss, self).__init__()
#         self.weight_focal = weight_focal
#         self.weight_dice = weight_dice
#         self.smooth = smooth
#         self.num_classes = num_classes
#         self.aux_weights = aux_weights
#         self.ignore_index = ignore_index
        
#         # Load and register class weights
#         if isinstance(class_weights, str) and os.path.isfile(class_weights):
#             class_weights = self._load_class_weights_from_ini(class_weights)
#             print(f"[ImprovedLoss] Loaded class weights: {class_weights}")
        
#         if isinstance(class_weights, (np.ndarray, list)):
#             class_weights = torch.tensor(class_weights, dtype=torch.float32)
        
#         if class_weights is not None:
#             self.register_buffer('class_weights', class_weights)
#         else:
#             self.class_weights = None
            
#         # 1. Use Focal Loss instead of CE
#         self.focal = FocalLoss(
#             alpha=focal_alpha, 
#             gamma=focal_gamma, 
#             class_weights=self.class_weights,
#             ignore_index=self.ignore_index
#         )

#     def _load_class_weights_from_ini(self, weight_file):
#         import configparser
#         config = configparser.ConfigParser()
#         config.read(weight_file)
#         weights = np.zeros(self.num_classes)
#         try:
#             # --- NEW CART Weights ---
#             weights[0] = float(config['ClassWeights']['bare_ground'])
#             weights[1] = float(config['ClassWeights']['rocky_terrain'])
#             weights[2] = float(config['ClassWeights']['devel_struct'])
#             weights[3] = float(config['ClassWeights']['road'])
#             weights[4] = float(config['ClassWeights']['shrubs'])
#             weights[5] = float(config['ClassWeights']['trees'])
#             weights[6] = float(config['ClassWeights']['sky'])
#             weights[7] = float(config['ClassWeights']['water'])
#             weights[8] = float(config['ClassWeights']['vehicles'])
#             weights[9] = float(config['ClassWeights']['person'])

#         except KeyError as e:
#             print(f"Key Error loading weights: {e}")
#             print("Assuming default ones (1.0) or check your .ini file keys.")
#             return np.ones(self.num_classes)
#         except Exception as e:
#             raise ValueError(f"Error loading {weight_file}: {e}")
            
#         return weights
    
#     def dice_loss(self, pred, target):
#         """
#         2. More Stable Dice Loss
#         Calculates intersection and union over the whole batch for each class.
#         Ignores pixels labeled with ignore_index.
#         """
#         pred_soft = F.softmax(pred, dim=1)
        
#         # Handle ignore_index for One-Hot encoding
#         valid_mask = target != self.ignore_index
#         target_safe = target.clone()
#         target_safe[~valid_mask] = 0 # Map ignored to 0 temporarily to avoid crash
        
#         # Create one-hot
#         target_onehot = F.one_hot(target_safe, self.num_classes).permute(0, 3, 1, 2).float()
        
#         # Zero out the one-hot vectors where the pixel was ignored
#         # valid_mask is [B, H, W] -> unsqueeze to [B, 1, H, W] -> expand to [B, C, H, W]
#         mask_expanded = valid_mask.unsqueeze(1).expand_as(target_onehot)
        
#         # --- CRITICAL FIX START ---
#         # We must mask BOTH the target AND the prediction.
#         # If we don't mask pred_soft, the model's predictions in the ignored (background)
#         # area will contribute to the Union denominator, but not the Intersection numerator.
#         # This forces the Dice score to ~0.
        
#         target_onehot = target_onehot * mask_expanded.float()
#         pred_soft_masked = pred_soft * mask_expanded.float()
        
#         # Sum over (Batch, H, W) dims
#         intersection = (pred_soft_masked * target_onehot).sum(dim=(0, 2, 3))
#         union = pred_soft_masked.sum(dim=(0, 2, 3)) + target_onehot.sum(dim=(0, 2, 3))
#         # --- CRITICAL FIX END ---
        
#         # We need to ensure we don't divide by zero if a class is not present
#         dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
#         # Average the loss over all classes
#         return (1 - dice).mean()

#     def _compute_loss(self, pred, target):
#         """Computes the combined loss for a single prediction."""
#         loss_focal = self.focal(pred, target)
#         loss_dice = self.dice_loss(pred, target)
#         return self.weight_focal * loss_focal + self.weight_dice * loss_dice

#     def _remap_cart_labels(self, target):
#         """
#         Remaps raw CART dataset labels to model training labels.
#         Raw: 0=Unknown, 1=Background, 2..11=Classes
#         Target: 0,1 -> Ignore(255); 2..11 -> 0..9
#         """
#         # Create a new tensor filled with ignore_index
#         new_target = torch.full_like(target, self.ignore_index)
        
#         # Identify valid class pixels (raw IDs 2 to 11)
#         valid_mask = (target >= 2) & (target <= 11)
        
#         # Shift valid pixels by -2 (2->0, 3->1, ... 11->9)
#         new_target[valid_mask] = target[valid_mask] - 2
        
#         return new_target

#     def forward(self, prediction, target):
#         """
#         prediction: Tuple (main_output, aux_outputs_dict)
#         target: [B, H, W] ground truth
#         """
#         # Unwrap tuple if it comes from a model returning (out, aux)
        
#         if isinstance(prediction, tuple):
#             main_pred = prediction[0]
#             aux_outputs = prediction[1]
#         else:
#             main_pred = prediction
#             aux_outputs = None

#         print(main_pred.shape, target.shape)

#         # Ensure target shape
#         if target.ndim == 4:
#             target = target.squeeze(1)
#         if target.dtype != torch.long:
#             target = target.long()

#         # --- FIX: Apply Label Mapping for CART ---
#         # Remap raw labels (0-11) to valid class indices (0-9) and ignore index
#         target = self._remap_cart_labels(target)
        
#         # Main prediction loss
#         main_loss = self._compute_loss(main_pred, target)
        
#         # 3. Explicitly Weighted Deep Supervision
#         if aux_outputs is not None and len(aux_outputs) > 0 and isinstance(aux_outputs, dict):
#             aux_loss = 0
#             for aux_name, aux_pred in aux_outputs.items():
#                 if aux_name in self.aux_weights:
#                     # Resize target to match aux prediction
#                     # Use float target for interpolation, then cast back to long
#                     # Note: Interpolation of labels should be Nearest Neighbor
#                     target_aux = F.interpolate(
#                         target.unsqueeze(1).float(),
#                         size=aux_pred.shape[2:],
#                         mode='nearest'
#                     ).squeeze(1).long()
                    
#                     aux_loss += self.aux_weights[aux_name] * self._compute_loss(aux_pred, target_aux)
            
#             total_loss = main_loss + aux_loss
#         else:
#             total_loss = main_loss
        
#         return total_loss
# SEG CART




    
class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, target, prediction):
        diff = prediction - target
        loss = torch.sqrt(diff ** 2 + self.epsilon ** 2)
        return torch.mean(loss)
    


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        self.loss_model = HybridLoss_SEG()

    def forward(self, y_pred, y_true, weight=None, **kwargs):
        return self.loss_model(y_true, y_pred)
        # return l1_loss(y_pred, y_true)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

# def gradient(input_tensor, direction):
#     smooth_kernel_x = torch.reshape(torch.tensor([[0, 0], [-1, 1]], dtype=torch.float32), [2, 2, 1, 1])
#     smooth_kernel_y = torch.transpose(smooth_kernel_x, 0, 1)
#     if direction == "x":
#         kernel = smooth_kernel_x
#     elif direction == "y":
#         kernel = smooth_kernel_y
#     gradient_orig = torch.abs(torch.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))
#     grad_min = torch.min(gradient_orig)
#     grad_max = torch.max(gradient_orig)
#     grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
#     return grad_norm

# class SmoothLoss(nn.Moudle):
#     """ illumination smoothness"""

#     def __init__(self, loss_weight=0.15, reduction='mean', eps=1e-2):
#         super(SmoothLoss,self).__init__()
#         self.loss_weight = loss_weight
#         self.eps = eps
#         self.reduction = reduction
    
#     def forward(self, illu, img):
#         # illu: b×c×h×w   illumination map
#         # img:  b×c×h×w   input image
#         illu_gradient_x = gradient(illu, "x")
#         img_gradient_x  = gradient(img, "x")
#         x_loss = torch.abs(torch.div(illu_gradient_x, torch.maximum(img_gradient_x, 0.01)))

#         illu_gradient_y = gradient(illu, "y")
#         img_gradient_y  = gradient(img, "y")
#         y_loss = torch.abs(torch.div(illu_gradient_y, torch.maximum(img_gradient_y, 0.01)))

#         loss = torch.mean(x_loss + y_loss) * self.loss_weight

#         return loss

# class MultualLoss(nn.Moudle):
#     """ Multual Consistency"""

#     def __init__(self, loss_weight=0.20, reduction='mean'):
#         super(MultualLoss,self).__init__()

#         self.loss_weight = loss_weight
#         self.reduction = reduction
    

#     def forward(self, illu):
#         # illu: b x c x h x w
#         gradient_x = gradient(illu,"x")
#         gradient_y = gradient(illu,"y")

#         x_loss = gradient_x * torch.exp(-10*gradient_x)
#         y_loss = gradient_y * torch.exp(-10*gradient_y)

#         loss = torch.mean(x_loss+y_loss) * self.loss_weight
#         return loss




