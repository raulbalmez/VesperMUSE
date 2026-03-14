import os
import configparser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

def multiscale_ssim_loss(y_true, y_pred, max_val=1.0, power_factors=[0.5, 0.5]):
    return 1.0 - ms_ssim(y_true, y_pred, data_range=max_val, size_average=True)



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



class HybridLoss_LLIE(nn.Module):
    def __init__(self, segmentation_feature_extractor=None, device=None, 
                 alpha1=1.0, alpha2=0.01, alpha3=0.2, alpha4=0.3):
        """
        Args:
            segmentation_feature_extractor: A pretrained segmentation encoder (e.g., ResNet backbone).
                                            Must return feature maps or logits.
        """
        super(HybridLoss_LLIE, self).__init__()
        
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
        print("ENH LOSS")
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
import numpy as np
import os
import configparser

class PST900_Loss(nn.Module):
    """
    Loss specifically for PST900 Dataset (5 classes).
    Combined Focal + Dice loss with explicit deep supervision weights.
    """
    def __init__(self, 
                 weight_focal=2.0, 
                 weight_dice=1.5, 
                 smooth=1.0,
                 class_weights="/home/ancuti/expendables/VesperMUSE/data/PST900_RGBT_Dataset/weights.ini", 
                 num_classes=5, 
                 focal_gamma=2.5,
                 focal_alpha=0.25,
                 # 🔹 FIX: Updated keys to match VesperMUSE_General outputs
                 aux_weights={'stage3': 0.7, 'stage2': 0.3}):
        
        super(PST900_Loss, self).__init__()
        self.weight_focal = weight_focal
        self.weight_dice = weight_dice
        self.smooth = smooth
        self.num_classes = num_classes
        self.aux_weights = aux_weights
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        
        if isinstance(class_weights, str) and os.path.isfile(class_weights):
            weights_np = self._load_class_weights_from_ini(class_weights)
            print(f"[PST900Loss] Loaded class weights: {weights_np}")
            self.register_buffer('class_weights', torch.tensor(weights_np, dtype=torch.float32))
        elif isinstance(class_weights, (np.ndarray, list)):
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def _load_class_weights_from_ini(self, weight_file):
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
            raise ValueError(f"Invalid or missing entries in {weight_file} for PST900")
        return weights

    def compute_focal_loss(self, pred, target):
        # 1. CE Loss handles the .ini class weights
        ce_loss = F.cross_entropy(pred, target, reduction='none', weight=self.class_weights)
        
        pred_soft = F.softmax(pred, dim=1)
        p = torch.gather(pred_soft, 1, target.unsqueeze(1)).squeeze(1)
        
        # 2. Modulating factor
        modulating_factor = (1 - p) ** self.focal_gamma
        
        # 🔹 FIX: Removed manual Alpha binary weighting.
        # This ensures the model follows your specific .ini weights for all 5 classes.
        focal_loss = modulating_factor * ce_loss
        return focal_loss.mean()

    def compute_dice_loss(self, pred, target):
        pred_soft = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        
        intersection = (pred_soft * target_onehot).sum(dim=(0, 2, 3))
        union = pred_soft.sum(dim=(0, 2, 3)) + target_onehot.sum(dim=(0, 2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice

        # 🔹 FIX: Apply Class Weights to Dice loss
        if self.class_weights is not None:
            dice_loss = dice_loss * self.class_weights
            
        # 🔹 OPTIMIZATION: Mean over classes actually present in batch
        classes_present = (target_onehot.sum(dim=(0, 2, 3)) > 0).float()
        return dice_loss.sum() / (classes_present.sum() + 1e-6)

    def _compute_combined_loss(self, pred, target):
        loss_focal = self.compute_focal_loss(pred, target)
        loss_dice = self.compute_dice_loss(pred, target)
        return self.weight_focal * loss_focal + self.weight_dice * loss_dice

    def forward(self, prediction, target):
        if isinstance(prediction, tuple):
            main_pred = prediction[0]
            aux_outputs = prediction[1]
        else:
            main_pred = prediction
            aux_outputs = None

        if target.ndim == 4: target = target.squeeze(1)
        if target.dtype != torch.long: target = target.long()
        
        main_loss = self._compute_combined_loss(main_pred, target)
        
        if aux_outputs is not None and len(aux_outputs) > 0 and isinstance(aux_outputs, dict):
            aux_loss = 0
            for aux_name, aux_pred in aux_outputs.items():
                if aux_name in self.aux_weights:
                    target_aux = F.interpolate(
                        target.unsqueeze(1).float(),
                        size=aux_pred.shape[2:],
                        mode='nearest'
                    ).squeeze(1).long()
                    aux_loss += self.aux_weights[aux_name] * self._compute_combined_loss(aux_pred, target_aux)
            return main_loss + aux_loss
        
        return main_loss


class MFNet_Loss(nn.Module):
    """
    Loss specifically for MFNet Dataset (9 classes).
    Combined Focal + Dice loss with explicit deep supervision weights.
    """
    def __init__(self, 
                 weight_focal=2.0, 
                 weight_dice=1.5, 
                 smooth=1.0,
                 class_weights="/home/ancuti/expendables/VesperMUSE/data/MFNet_Dataset/weights.ini", 
                 num_classes=9, 
                 focal_gamma=2.5,
                 focal_alpha=0.25,
                 aux_weights={'stage1': 0.7, 'stage2': 0.3}):
        
        super(MFNet_Loss, self).__init__()
        self.weight_focal = weight_focal
        self.weight_dice = weight_dice
        self.smooth = smooth
        self.num_classes = num_classes
        self.aux_weights = aux_weights
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        
        if isinstance(class_weights, str) and os.path.isfile(class_weights):
            weights_np = self._load_class_weights_from_ini(class_weights)
            print(f"[MFNetLoss] Loaded class weights: {weights_np}")
            self.register_buffer('class_weights', torch.tensor(weights_np, dtype=torch.float32))
        elif isinstance(class_weights, (np.ndarray, list)):
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def _load_class_weights_from_ini(self, weight_file):
        config = configparser.ConfigParser()
        config.read(weight_file)
        weights = np.zeros(self.num_classes)
        try:
            weights[0] = float(config['ClassWeights']['background'])
            weights[1] = float(config['ClassWeights']['car'])
            weights[2] = float(config['ClassWeights']['person'])
            weights[3] = float(config['ClassWeights']['bike'])
            weights[4] = float(config['ClassWeights']['curve'])
            weights[5] = float(config['ClassWeights']['car stop'])
            weights[6] = float(config['ClassWeights']['guardrail'])
            weights[7] = float(config['ClassWeights']['color cone'])
            weights[8] = float(config['ClassWeights']['bump'])
        except KeyError:
            # Fallback or strict error depending on preference, currently strict to match PST900 style
            raise ValueError(f"Invalid or missing entries in {weight_file} for MFNet")
        return weights

    def compute_focal_loss(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none', weight=self.class_weights)
        pred_soft = F.softmax(pred, dim=1)
        p = torch.gather(pred_soft, 1, target.unsqueeze(1)).squeeze(1)
        modulating_factor = (1 - p) ** self.focal_gamma
        
        if self.focal_alpha is not None:
            alpha_weight = target.float().new_tensor([self.focal_alpha]).expand_as(target)
            alpha_weight = torch.where(target > 0, alpha_weight, 1.0 - alpha_weight)
            modulating_factor = alpha_weight * modulating_factor

        focal_loss = modulating_factor * ce_loss
        return focal_loss.mean()

    def compute_dice_loss(self, pred, target):
        pred_soft = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        intersection = (pred_soft * target_onehot).sum(dim=(0, 2, 3))
        union = pred_soft.sum(dim=(0, 2, 3)) + target_onehot.sum(dim=(0, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return (1 - dice).mean()

    def _compute_combined_loss(self, pred, target):
        loss_focal = self.compute_focal_loss(pred, target)
        loss_dice = self.compute_dice_loss(pred, target)
        return self.weight_focal * loss_focal + self.weight_dice * loss_dice

    def forward(self, prediction, target):
        if isinstance(prediction, tuple):
            main_pred = prediction[0]
            aux_outputs = prediction[1]
        else:
            main_pred = prediction
            aux_outputs = None

        if target.ndim == 4: target = target.squeeze(1)
        if target.dtype != torch.long: target = target.long()
        
        main_loss = self._compute_combined_loss(main_pred, target)
        
        if aux_outputs is not None and len(aux_outputs) > 0:
            aux_loss = 0
            for aux_name, aux_pred in aux_outputs.items():
                if aux_name in self.aux_weights:
                    target_aux = F.interpolate(
                        target.unsqueeze(1).float(),
                        size=aux_pred.shape[2:],
                        mode='nearest'
                    ).squeeze(1).long()
                    aux_loss += self.aux_weights[aux_name] * self._compute_combined_loss(aux_pred, target_aux)
            return main_loss + aux_loss
        
        return main_loss


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import configparser

class CART_Loss(nn.Module):
    """
    Loss specifically for CART Dataset.
    Handles label remapping (shift -2) and ignore_index (255).
    """
    def __init__(self, 
                 weight_focal=2.0, 
                 weight_dice=1.5, 
                 smooth=1.0,
                 class_weights="/home/ancuti/expendables/VesperMUSE/data/CART/cart_weights.ini", 
                 num_classes=10, 
                 focal_gamma=2.5,
                 focal_alpha=0.25,
                 ignore_index=255,
                 # 🔹 FIX: Changed keys to match VesperMUSE_General outputs (stage3, stage2)
                 aux_weights={'stage3': 0.7, 'stage2': 0.3}):
        
        super(CART_Loss, self).__init__()
        self.weight_focal = weight_focal
        self.weight_dice = weight_dice
        self.smooth = smooth
        self.num_classes = num_classes
        self.aux_weights = aux_weights
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.ignore_index = ignore_index
        
        if isinstance(class_weights, str) and os.path.isfile(class_weights):
            weights_np = self._load_class_weights_from_ini(class_weights)
            print(f"[CARTLoss] Loaded class weights: {weights_np}")
            self.register_buffer('class_weights', torch.tensor(weights_np, dtype=torch.float32))
        elif isinstance(class_weights, (np.ndarray, list)):
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def _load_class_weights_from_ini(self, weight_file):
        config = configparser.ConfigParser()
        config.read(weight_file)
        weights = np.zeros(self.num_classes)
        try:
            weights[0] = float(config['ClassWeights']['bare_ground'])
            weights[1] = float(config['ClassWeights']['rocky_terrain'])
            weights[2] = float(config['ClassWeights']['devel_struct'])
            weights[3] = float(config['ClassWeights']['road'])
            weights[4] = float(config['ClassWeights']['shrubs'])
            weights[5] = float(config['ClassWeights']['trees'])
            weights[6] = float(config['ClassWeights']['sky'])
            weights[7] = float(config['ClassWeights']['water'])
            weights[8] = float(config['ClassWeights']['vehicles'])
            weights[9] = float(config['ClassWeights']['person'])
        except KeyError as e:
            print(f"Key Error loading weights: {e}")
            return np.ones(self.num_classes)
        return weights

    def compute_focal_loss(self, pred, target):
        # 1. CE Loss (handles ignore_index and the .ini class weights)
        ce_loss = F.cross_entropy(
            pred, 
            target, 
            reduction='none', 
            weight=self.class_weights, 
            ignore_index=self.ignore_index
        )
        
        # 2. Probability of correct class
        pred_soft = F.softmax(pred, dim=1)
        valid_mask = target != self.ignore_index
        target_safe = target.clone()
        target_safe[~valid_mask] = 0  
        
        p = torch.gather(pred_soft, 1, target_safe.unsqueeze(1)).squeeze(1)
        
        # 3. Modulating factor (Gamma handles the "Hard Example" mining)
        modulating_factor = (1 - p) ** self.focal_gamma
        
        # 🔹 FIX: Removed manual Alpha binary weighting. 
        # Since self.class_weights are already in the ce_loss, re-applying 
        # focal_alpha as a hard toggle (target > 0) was diluting your .ini weights.
        focal_loss = modulating_factor * ce_loss
        
        # Zero out loss for ignored pixels
        focal_loss = torch.where(valid_mask, focal_loss, torch.zeros_like(focal_loss))
        
        # Normalize by valid pixels
        return focal_loss.sum() / (valid_mask.sum() + 1e-6)

    def compute_dice_loss(self, pred, target):
        pred_soft = F.softmax(pred, dim=1)
        
        valid_mask = target != self.ignore_index
        target_safe = target.clone()
        target_safe[~valid_mask] = 0 
        
        target_onehot = F.one_hot(target_safe, self.num_classes).permute(0, 3, 1, 2).float()
        mask_expanded = valid_mask.unsqueeze(1).expand_as(target_onehot)
        
        target_onehot = target_onehot * mask_expanded.float()
        pred_soft_masked = pred_soft * mask_expanded.float()
        
        intersection = (pred_soft_masked * target_onehot).sum(dim=(0, 2, 3))
        union = pred_soft_masked.sum(dim=(0, 2, 3)) + target_onehot.sum(dim=(0, 2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice

        # 🔹 FIX: Apply Class Weights to Dice loss. 
        # This prevents the "Average" Dice from ignoring the Vehicles/Person classes.
        if self.class_weights is not None:
            dice_loss = dice_loss * self.class_weights
            
        return dice_loss.mean()

    def _remap_cart_labels(self, target):
        """Remaps raw CART labels (2..11) to (0..9) and others to ignore_index."""
        new_target = torch.full_like(target, self.ignore_index)
        valid_mask = (target >= 2) & (target <= 11)
        new_target[valid_mask] = target[valid_mask] - 2
        return new_target

    def _compute_combined_loss(self, pred, target):
        loss_focal = self.compute_focal_loss(pred, target)
        loss_dice = self.compute_dice_loss(pred, target)
        return self.weight_focal * loss_focal + self.weight_dice * loss_dice

    def forward(self, prediction, target):
        if isinstance(prediction, tuple):
            main_pred = prediction[0]
            aux_outputs = prediction[1]
        else:
            main_pred = prediction
            aux_outputs = None

        if target.ndim == 4: target = target.squeeze(1)
        if target.dtype != torch.long: target = target.long()

        # Apply CART specific remapping
        target = self._remap_cart_labels(target)
        
        main_loss = self._compute_combined_loss(main_pred, target)
        
        if aux_outputs is not None and len(aux_outputs) > 0 and isinstance(aux_outputs, dict):
            aux_loss = 0
            for aux_name, aux_pred in aux_outputs.items():
                if aux_name in self.aux_weights:
                    target_aux = F.interpolate(
                        target.unsqueeze(1).float(),
                        size=aux_pred.shape[2:],
                        mode='nearest'
                    ).squeeze(1).long()
                    aux_loss += self.aux_weights[aux_name] * self._compute_combined_loss(aux_pred, target_aux)
            return main_loss + aux_loss
        
        return main_loss
