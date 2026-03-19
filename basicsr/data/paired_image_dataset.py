# from torch.utils import data as data
# from torchvision.transforms.functional import normalize

# from basicsr.data.data_util import (paired_paths_from_folder,
#                                     paired_DP_paths_from_folder,
#                                     paired_paths_from_lmdb,
#                                     paired_paths_from_meta_info_file)
# from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation
# from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP
# import basicsr.data.util as util

# import random
# import numpy as np
# import torch
# import cv2
# from pdb import set_trace as stx
# import os

# # ENH Dataloader
# class Dataset_PairedImage_Enhancement(data.Dataset):
#     """Paired image dataset for RGB + Thermal image restoration."""

#     def __init__(self, opt):
#         super(Dataset_PairedImage_Enhancement, self).__init__()
#         self.opt = opt
#         self.file_client = None
#         self.io_backend_opt = opt['io_backend']

#         self.mean = opt.get('mean', None)
#         self.std = opt.get('std', None)

#         # main folders
#         self.gt_folder = opt['dataroot_gt']
#         self.lq_folder = opt['dataroot_lq']

#         # optional thermal/second modality folder
#         self.second_folder = opt.get('lq_second', None)

#         self.filename_tmpl = opt.get('filename_tmpl', '{}')

#         # collect paths
#         if self.io_backend_opt['type'] == 'lmdb':
#             self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
#             self.io_backend_opt['client_keys'] = ['lq', 'gt']
#             self.paths = paired_paths_from_lmdb(
#                 [self.lq_folder, self.gt_folder], ['lq', 'gt'])
#         elif 'meta_info_file' in opt and opt['meta_info_file'] is not None:
#             self.paths = paired_paths_from_meta_info_file(
#                 [self.lq_folder, self.gt_folder], ['lq', 'gt'],
#                 opt['meta_info_file'], self.filename_tmpl)
#         else:
#             self.paths = paired_paths_from_folder(
#                 [self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

#         if opt['phase'] == 'train':
#             self.geometric_augs = opt.get('geometric_augs', False)

#     def __getitem__(self, index):
#         if self.file_client is None:
#             self.file_client = FileClient(self.io_backend_opt.pop('type'),
#                                           **self.io_backend_opt)

#         scale = self.opt['scale']
#         index = index % len(self.paths)

#         # --- Load GT ---
#         gt_path = self.paths[index]['gt_path']
#         img_bytes = self.file_client.get(gt_path, 'gt')
#         # img_gt = imfrombytes(img_bytes, float32=True, flag="grayscale")       # SEG_CHANGE
#         img_gt = imfrombytes(img_bytes, float32=True)
#         if img_gt.ndim == 2:
#             img_gt = np.expand_dims(img_gt, 2)

#         # --- Load RGB LQ ---
#         lq_path = self.paths[index]['lq_path']
#         img_bytes = self.file_client.get(lq_path, 'lq')
#         img_lq = imfrombytes(img_bytes, float32=True)

#         # --- Load thermal LQ (optional) ---
#         img_thermal = None
#         if self.second_folder is not None:
#             thermal_path = lq_path.replace(self.lq_folder, self.second_folder)
#             if os.path.exists(thermal_path):
#                 try:
#                     img_bytes = self.file_client.get(thermal_path, 'lq')
#                     img_thermal = imfrombytes(img_bytes, float32=True, flag='grayscale')
#                     if img_thermal.ndim == 2:
#                         img_thermal = np.expand_dims(img_thermal, 2)
#                 except Exception as e:
#                     print(f"Warning: Failed to load thermal image from {thermal_path}: {e}")
#                     img_thermal = np.zeros((img_lq.shape[0], img_lq.shape[1], 1), np.float32)
#             else:
#                 img_thermal = np.zeros((img_lq.shape[0], img_lq.shape[1], 1), np.float32)
#         else:
#             img_thermal = np.zeros((img_lq.shape[0], img_lq.shape[1], 1), np.float32)

#         # --- Data augmentations ---
#         if self.opt['phase'] == 'train':
#             gt_size = self.opt['gt_size']

#             # padding
#             img_gt, img_lq = padding(img_gt, img_lq, gt_size)
#             img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)

#             # crop the same region for thermal
#             h, w, _ = img_lq.shape
#             img_thermal = img_thermal[:h, :w, :]

#             # flip, rotation augmentations
#             if self.geometric_augs:
#                 img_gt, img_lq = random_augmentation(img_gt, img_lq)
#                 img_thermal, _ = random_augmentation(img_thermal, img_lq)

#         # --- Convert to tensor ---
#         img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
#         img_thermal = img2tensor([img_thermal], bgr2rgb=False, float32=True)[0]  # [1,H,W]

#         # --- Concatenate to get [4,H,W] ---
#         img_lq = torch.cat([img_lq, img_thermal], dim=0)

#         # --- Normalize ---
#         if self.mean is not None and self.std is not None:
#             rgb_mean = self.mean
#             rgb_std = self.std
#             t_mean = self.mean[:1]
#             t_std = self.std[:1]
#             normalize(img_lq, rgb_mean + t_mean, rgb_std + t_std, inplace=True)
#             normalize(img_gt, rgb_mean, rgb_std, inplace=True)
#         # print("Load: ", img_lq.shape, img_gt.shape, lq_path, gt_path)
#         # print("True: ", img_gt.shape, gt_path)
#         # print("Input: ", img_lq.shape, lq_path)

#         return {
#             'lq': img_lq,  # [4,H,W]
#             'gt': img_gt,
#             'lq_path': lq_path,
#             'gt_path': gt_path
#         }

#     def __len__(self):
#         return len(self.paths)


# from PIL import Image
# import numpy as np
# import torch
# import torch.utils.data as data
# import os


# import os.path as osp


# import os
# import os.path as osp
# import torch
# import torch.utils.data as data
# import torch.nn.functional as F
# import basicsr.data.util as util


# import torch
# import torch.utils.data as data
# import os
# import os.path as osp
# import numpy as np
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import basicsr.data.util as util


# import torch
# import torch.utils.data as data
# import os
# import os.path as osp
# import numpy as np
# import albumentations as A
# from basicsr.data import util  # Assuming 'util' is from basicsr or a similar library

# # Make sure to import any missing libraries (like os, np, torch, etc.)
# # The original code snippet was missing some imports.

# import os
# import os.path as osp
# import torch
# import torch.utils.data as data
# import torch.nn.functional as F
# import torchvision.transforms.functional as TF # <--- ADDED THIS
# import numpy as np
# import random
# import albumentations as A
# # Assuming 'util' is your custom utility script
# # import utils.util as util 
# # SEG Crop
# class Dataset_PairedImage_Segmentation(data.Dataset):
#     """Simplified paired RGB+Thermal dataset with GT labels (Torch only).""" 
#     def __init__(self, opt):
#         super(Dataset_PairedImage_Segmentation, self).__init__()
#         self.opt = opt
#         self.cache_data = opt.get('cache_data', False)
#         self.GT_root = opt['dataroot_gt']
#         self.LQ_root = opt['dataroot_lq']
#         self.thermal_root = opt.get('lq_second', None)
#         self.io_backend_opt = opt['io_backend']

#         # collect paths
#         self.data_info = {'path_LQ': [], 'path_GT': [], 'folder': [], 'idx': []}
#         self.imgs_LQ, self.imgs_GT = {}, {}

#         # Assuming util.glob_file_list exists
#         img_paths_LQ = util.glob_file_list(self.LQ_root)
#         img_paths_GT = util.glob_file_list(self.GT_root)
        
#         # Sort to ensure alignment
#         img_paths_LQ = sorted(img_paths_LQ)
#         img_paths_GT = sorted(img_paths_GT)

#         assert len(img_paths_LQ) == len(img_paths_GT), \
#             f"LQ ({len(img_paths_LQ)}) and GT ({len(img_paths_GT)}) counts mismatch"

#         folder_name = osp.basename(self.GT_root.rstrip('/'))
#         self.data_info['path_LQ'].extend(img_paths_LQ)
#         self.data_info['path_GT'].extend(img_paths_GT)
#         self.data_info['folder'].extend([folder_name] * len(img_paths_LQ))
#         self.data_info['idx'].extend([f"{i}/{len(img_paths_LQ)}" for i in range(len(img_paths_LQ))])

#         if self.cache_data:
#             self.imgs_LQ[folder_name] = img_paths_LQ
#             self.imgs_GT[folder_name] = img_paths_GT

#         # Color Augmentations (RGB Only) - Albumentations
#         self.rgb_color_aug = A.Compose([
#             A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8),
#             A.RandomGamma(gamma_limit=(80, 120), p=0.5),
#         ])

#     def __getitem__(self, index):
#         folder = self.data_info['folder'][index]
#         img_LQ_path = [self.data_info['path_LQ'][index]]
#         img_GT_path = [self.data_info['path_GT'][index]]

#         # ---- Load RGB LQ and GT ----
#         # util.read_img_seq2 returns Tensor [C,H,W] in range [0,1]
#         img_LQ = util.read_img_seq2(img_LQ_path, self.opt.get('train_size'))[0]  
        
#         # GT is usually raw integer indices
#         img_GT = util.read_img_seq2_raw(img_GT_path, self.opt.get('train_size'))[0] 

#         if img_GT.ndim == 3 and img_GT.shape[0] > 1:
#             img_GT = img_GT[0:1]
#         img_GT = img_GT.squeeze(0).long()  # [H,W]

#         # ---- Load Thermal ----
#         if self.thermal_root:
#             thermal_path = img_LQ_path[0].replace(self.LQ_root, self.thermal_root)
#             # Handle potential extension mismatch (jpg vs png) if necessary
#             # thermal_path = os.path.splitext(thermal_path)[0] + '.png' 
            
#             if os.path.exists(thermal_path):
#                 img_thermal = util.read_img_seq2([thermal_path], self.opt.get('train_size'))[0]
#                 if img_thermal.ndim == 2:
#                     img_thermal = img_thermal.unsqueeze(0)
#                 elif img_thermal.shape[0] > 1:
#                     # Convert 3-channel thermal to 1-channel if needed
#                     img_thermal = img_thermal.mean(dim=0, keepdim=True)
#             else:
#                 img_thermal = torch.zeros((1, *img_LQ.shape[1:]), dtype=torch.float32)
#         else:
#             img_thermal = torch.zeros((1, *img_LQ.shape[1:]), dtype=torch.float32)

#         # -------------------------------------------------------
#         # 3. Random Crop (Applies to RGB, Thermal, and GT)
#         # -------------------------------------------------------
#         crop_size = self.opt.get('random_crop', None)
        
#         if self.opt.get('phase') == 'train' and crop_size is not None:
#             if isinstance(crop_size, int):
#                 crop_h, crop_w = crop_size, crop_size
#             else:
#                 crop_h, crop_w = crop_size[0], crop_size[1]

#             H, W = img_LQ.shape[-2:]
            
#             if H <= crop_h: h_start = 0
#             else: h_start = random.randint(0, H - crop_h)
                
#             if W <= crop_w: w_start = 0
#             else: w_start = random.randint(0, W - crop_w)

#             h_end = h_start + crop_h
#             w_end = w_start + crop_w

#             img_LQ = img_LQ[:, h_start:h_end, w_start:w_end]
#             img_thermal = img_thermal[:, h_start:h_end, w_start:w_end]
#             img_GT = img_GT[h_start:h_end, w_start:w_end]

#         # -------------------------------------------------------
#         # 3.5 Geometric Augmentations (Synchronized) <--- NEW
#         # -------------------------------------------------------
#         if self.opt.get('phase') == 'train':
            
#             # A. Random Horizontal Flip (Standard for street scenes)
#             if random.random() < 0.5:
#                 img_LQ = TF.hflip(img_LQ)
#                 img_thermal = TF.hflip(img_thermal)
#                 img_GT = TF.hflip(img_GT) # Works on [H, W] or [C, H, W]

#             # B. Random Vertical Flip (Optional - usually OFF for MFNet/PST900)
#             if self.opt.get('use_vflip', False) and random.random() < 0.5:
#                 img_LQ = TF.vflip(img_LQ)
#                 img_thermal = TF.vflip(img_thermal)
#                 img_GT = TF.vflip(img_GT)

#             # C. Random Rotation (Small angles)
#             # SOTA often rotates +/- 10 degrees to learn invariance
#             if self.opt.get('use_rot', False) and random.random() < 0.5:
#                 angle = random.uniform(-10, 10)
#                 img_LQ = TF.rotate(img_LQ, angle, interpolation=TF.InterpolationMode.BILINEAR)
#                 img_thermal = TF.rotate(img_thermal, angle, interpolation=TF.InterpolationMode.BILINEAR)
                
#                 # IMPORTANT: For GT (Labels), use NEAREST to preserve integers
#                 # TF.rotate expects [C, H, W], so unsqueeze GT [H, W] -> [1, H, W]
#                 img_GT = TF.rotate(img_GT.unsqueeze(0), angle, interpolation=TF.InterpolationMode.NEAREST).squeeze(0)

#         # -------------------------------------------------------
#         # 3.6 Color Augmentation (RGB ONLY)
#         # -------------------------------------------------------
#         # Uncomment to enable Albumentations
#         # if self.opt.get('phase') == 'train':
#         #     rgb_np = img_LQ.permute(1, 2, 0).numpy() * 255.0
#         #     rgb_np = rgb_np.astype(np.uint8)
#         #     augmented = self.rgb_color_aug(image=rgb_np)
#         #     rgb_aug = augmented['image']
#         #     img_LQ = torch.from_numpy(rgb_aug).permute(2, 0, 1).float() / 255.0

#         # ---- Concatenate RGB + Thermal ----
#         if img_LQ.shape[0] == 1:
#             img_LQ = img_LQ.repeat(3, 1, 1)
        
#         # Final concat: (4, H, W)
#         img_LQ = torch.cat([img_LQ, img_thermal], dim=0)

#         # ---- Normalize ----
#         mean, std = self.opt.get('mean', None), self.opt.get('std', None)
#         if mean is not None and std is not None:
#             # Add dummy 4th channel for Thermal normalization (usually (0.5,), (0.5,))
#             # Or extend RGB mean/std to cover thermal
#             if len(mean) == 3:
#                 mean_ext = mean + [0.5] # Assume 0.5 mean for thermal
#                 std_ext = std + [0.5]   # Assume 0.5 std for thermal
#             else:
#                 mean_ext = mean
#                 std_ext = std
            
#             mean_t = torch.tensor(mean_ext, dtype=img_LQ.dtype, device=img_LQ.device)[:, None, None]
#             std_t = torch.tensor(std_ext, dtype=img_LQ.dtype, device=img_LQ.device)[:, None, None]
            
#             img_LQ = (img_LQ - mean_t) / std_t

#         return {
#             'lq': img_LQ.contiguous(),    
#             'gt': img_GT.contiguous(),    
#             'folder': folder,
#             'idx': self.data_info['idx'][index],
#             'lq_path': img_LQ_path[0],
#             'gt_path': img_GT_path[0]
#         }

#     def __len__(self):
#         return len(self.data_info['path_LQ'])
# # SEG Crop


import os
import random
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import normalize

# BasicSR and Imaging Utilities
import basicsr.data.util as util
from basicsr.data.data_util import (
    paired_paths_from_folder,
    paired_paths_from_lmdb,
    paired_paths_from_meta_info_file
)
from basicsr.data.transforms import augment, paired_random_crop, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding

# Optional Augmentation
import albumentations as A

---

class Dataset_PairedImage_Enhancement(data.Dataset):
    """Dataset for RGB + Thermal image restoration/enhancement."""

    def __init__(self, opt):
        super(Dataset_PairedImage_Enhancement, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)

        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        self.second_folder = opt.get('lq_second', None)
        self.filename_tmpl = opt.get('filename_tmpl', '{}')

        if self.io_backend_opt['type'] == 'lmdb':
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif opt.get('meta_info_file'):
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

        self.geometric_augs = opt.get('geometric_augs', False) if opt['phase'] == 'train' else False

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)

        # Load GT
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        if img_gt.ndim == 2:
            img_gt = np.expand_dims(img_gt, 2)

        # Load RGB LQ
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # Load Thermal LQ
        if self.second_folder:
            thermal_path = lq_path.replace(self.lq_folder, self.second_folder)
            if os.path.exists(thermal_path):
                try:
                    t_bytes = self.file_client.get(thermal_path, 'lq')
                    img_thermal = imfrombytes(t_bytes, float32=True, flag='grayscale')
                    if img_thermal.ndim == 2:
                        img_thermal = np.expand_dims(img_thermal, 2)
                except Exception:
                    img_thermal = np.zeros((img_lq.shape[0], img_lq.shape[1], 1), np.float32)
            else:
                img_thermal = np.zeros((img_lq.shape[0], img_lq.shape[1], 1), np.float32)
        else:
            img_thermal = np.zeros((img_lq.shape[0], img_lq.shape[1], 1), np.float32)

        # Augmentations
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            
            h, w, _ = img_lq.shape
            img_thermal = img_thermal[:h, :w, :]

            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)
                img_thermal, _ = random_augmentation(img_thermal, img_lq)

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        img_thermal = img2tensor([img_thermal], bgr2rgb=False, float32=True)[0]

        img_lq = torch.cat([img_lq, img_thermal], dim=0)

        if self.mean is not None and self.std is not None:
            rgb_mean, rgb_std = self.mean, self.std
            t_mean, t_std = self.mean[:1], self.std[:1]
            normalize(img_lq, rgb_mean + t_mean, rgb_std + t_std, inplace=True)
            normalize(img_gt, rgb_mean, rgb_std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)

---

class Dataset_PairedImage_Segmentation(data.Dataset):
    """Dataset for RGB + Thermal Semantic Segmentation."""

    def __init__(self, opt):
        super(Dataset_PairedImage_Segmentation, self).__init__()
        self.opt = opt
        self.GT_root = opt['dataroot_gt']
        self.LQ_root = opt['dataroot_lq']
        self.thermal_root = opt.get('lq_second', None)

        img_paths_LQ = sorted(util.glob_file_list(self.LQ_root))
        img_paths_GT = sorted(util.glob_file_list(self.GT_root))
        
        assert len(img_paths_LQ) == len(img_paths_GT), "Mismatch between LQ and GT file counts."

        self.data_info = {
            'path_LQ': img_paths_LQ,
            'path_GT': img_paths_GT,
            'folder': [os.path.basename(self.GT_root.rstrip('/'))] * len(img_paths_LQ),
            'idx': [f"{i}/{len(img_paths_LQ)}" for i in range(len(img_paths_LQ))]
        }

    def __getitem__(self, index):
        lq_path = [self.data_info['path_LQ'][index]]
        gt_path = [self.data_info['path_GT'][index]]

        # Load Data
        img_lq = util.read_img_seq2(lq_path, self.opt.get('train_size'))[0]
        img_gt = util.read_img_seq2_raw(gt_path, self.opt.get('train_size'))[0]

        if img_gt.ndim == 3 and img_gt.shape[0] > 1:
            img_gt = img_gt[0:1]
        img_gt = img_gt.squeeze(0).long()

        if self.thermal_root:
            t_path = lq_path[0].replace(self.LQ_root, self.thermal_root)
            if os.path.exists(t_path):
                img_thermal = util.read_img_seq2([t_path], self.opt.get('train_size'))[0]
                if img_thermal.ndim == 2:
                    img_thermal = img_thermal.unsqueeze(0)
                elif img_thermal.shape[0] > 1:
                    img_thermal = img_thermal.mean(dim=0, keepdim=True)
            else:
                img_thermal = torch.zeros((1, *img_lq.shape[1:]), dtype=torch.float32)
        else:
            img_thermal = torch.zeros((1, *img_lq.shape[1:]), dtype=torch.float32)

        # Random Crop
        crop_size = self.opt.get('random_crop')
        if self.opt.get('phase') == 'train' and crop_size:
            h_size, w_size = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
            H, W = img_lq.shape[-2:]
            
            h_start = random.randint(0, max(0, H - h_size))
            w_start = random.randint(0, max(0, W - w_size))

            img_lq = img_lq[:, h_start:h_start+h_size, w_start:w_start+w_size]
            img_thermal = img_thermal[:, h_start:h_start+h_size, w_start:w_start+w_size]
            img_gt = img_gt[h_start:h_start+h_size, w_start:w_start+w_size]

        # Geometric Augmentations
        if self.opt.get('phase') == 'train':
            if random.random() < 0.5:
                img_lq, img_thermal, img_gt = TF.hflip(img_lq), TF.hflip(img_thermal), TF.hflip(img_gt)

            if self.opt.get('use_rot') and random.random() < 0.5:
                angle = random.uniform(-10, 10)
                img_lq = TF.rotate(img_lq, angle, interpolation=TF.InterpolationMode.BILINEAR)
                img_thermal = TF.rotate(img_thermal, angle, interpolation=TF.InterpolationMode.BILINEAR)
                img_gt = TF.rotate(img_gt.unsqueeze(0), angle, interpolation=TF.InterpolationMode.NEAREST).squeeze(0)

        if img_lq.shape[0] == 1:
            img_lq = img_lq.repeat(3, 1, 1)
        
        img_lq = torch.cat([img_lq, img_thermal], dim=0)

        # Normalization
        mean, std = self.opt.get('mean'), self.opt.get('std')
        if mean is not None and std is not None:
            mean_ext = mean + [0.5] if len(mean) == 3 else mean
            std_ext = std + [0.5] if len(std) == 3 else std
            
            m_t = torch.tensor(mean_ext, dtype=img_lq.dtype).view(-1, 1, 1)
            s_t = torch.tensor(std_ext, dtype=img_lq.dtype).view(-1, 1, 1)
            img_lq = (img_lq - m_t) / s_t

        return {
            'lq': img_lq.contiguous(),
            'gt': img_gt.contiguous(),
            'folder': self.data_info['folder'][index],
            'idx': self.data_info['idx'][index],
            'lq_path': lq_path[0],
            'gt_path': gt_path[0]
        }

    def __len__(self):
        return len(self.data_info['path_LQ'])