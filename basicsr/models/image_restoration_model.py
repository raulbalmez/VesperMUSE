import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import glob

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial

try :
    from torch.cuda.amp import autocast, GradScaler
    load_amp = True
except:
    load_amp = False


class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(
            torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1, 1)).item()

        r_index = torch.randperm(target.size(0)).to(self.device)

        target = lam * target + (1 - lam) * target[r_index, :]
        input_ = lam * input_ + (1 - lam) * input_[r_index, :]

        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments) - 1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_





class ImageCleanModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageCleanModel, self).__init__(opt)

        # define mixed precision
        self.use_amp = opt.get('use_amp', False) and load_amp
        self.amp_scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp:
            print('Using Automatic Mixed Precision')
        else:
            print('Not using Automatic Mixed Precision')
                  
        # define network
        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta = self.opt['train']['mixing_augs'].get(
                'mixup_beta', 1.2)
            use_identity = self.opt['train']['mixing_augs'].get(
                'use_identity', False)
            self.mixing_augmentation = Mixing_Augment(
                mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        # self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()



    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(
                self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)  #根据pop出来的loss_type找到对应的loss函数
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)      #如何写 weighted loss 呢？传参构造Loss函数
        else:
            raise ValueError('pixel loss are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params_cfg = train_opt['optim_g'].copy()
        
        logger = get_root_logger()
        
        # 1. Extract critical info
        optim_type = optim_params_cfg.pop('type')
        base_lr = optim_params_cfg.get('lr')
        
        # Remove global 'lr' from config so it doesn't conflict with groups
        if 'lr' in optim_params_cfg:
            optim_params_cfg.pop('lr')

        # 2. Define lists
        backbone_params = []
        head_params = []
        
        # DEBUG: Store names to print them later
        backbone_names_debug = []
        head_names_debug = []
        
        # Keywords for your backbones
        backbone_keywords = ['rgb_net', 'thm_net', 'dep_net']
        
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                # Check if name contains backbone keywords
                is_backbone = any(keyword in k for keyword in backbone_keywords)
                
                if is_backbone:
                    backbone_params.append(v)
                    backbone_names_debug.append(k)
                else:
                    head_params.append(v)
                    head_names_debug.append(k)
            else:
                logger.warning(f'Params {k} will not be optimized.')

        # 3. Create Groups
        optim_groups = [
            {
                'params': backbone_params,
                'lr': base_lr * 0.1,  # 10x Slower
                'name': 'backbone_group'
            },
            {
                'params': head_params,
                'lr': base_lr,        # Standard Speed
                'name': 'head_group'
            }
        ]

        # 4. Initialize Optimizer
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_groups, **optim_params_cfg)
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_groups, **optim_params_cfg)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')
            
        self.optimizers.append(self.optimizer_g)

        # ============================================================
        #                       DEBUGGING OUTPUT
        # ============================================================
        logger.info(f"{'='*20} OPTIMIZER SETUP DEBUG {'='*20}")
        
        # Group 0: Backbone
        lr_backbone = self.optimizer_g.param_groups[0]['lr']
        count_backbone = len(self.optimizer_g.param_groups[0]['params'])
        logger.info(f"[Group 0: Backbone] LR: {lr_backbone:.2e} | Params: {count_backbone}")
        logger.info(f"   First 5 Params in Backbone:")
        for name in backbone_names_debug[:5]:
            logger.info(f"     - {name}")

        # Group 1: Head
        lr_head = self.optimizer_g.param_groups[1]['lr']
        count_head = len(self.optimizer_g.param_groups[1]['params'])
        logger.info(f"[Group 1: Head]     LR: {lr_head:.2e} | Params: {count_head}")
        logger.info(f"   First 5 Params in Head:")
        for name in head_names_debug[:5]:
            logger.info(f"     - {name}")
            
        logger.info(f"{'='*62}")

    # def setup_optimizers(self):
    #     train_opt = self.opt['train']
    #     optim_g_opt = train_opt['optim_g']
        
    #     # --- NEW FIX ---
    #     # Pop the 'type' key. It's not an argument for the optimizer.
    #     optim_g_opt.pop('type')
    #     # --- END NEW FIX ---
        
    #     # 1. POP the paramwise_cfg from the optimizer options
    #     # It's not an argument for AdamW, so we handle it manually
    #     paramwise_cfg = optim_g_opt.pop('paramwise_cfg', None)
        
    #     param_groups = []
    #     if paramwise_cfg is None:
    #         # No custom settings, add all params with default LR
    #         param_groups.append({'params': self.net_g.parameters()})
    #     else:
    #         # 2. Build parameter groups manually
    #         base_lr = optim_g_opt['lr']
    #         custom_keys = paramwise_cfg['custom_keys']
            
    #         # Create a set of all parameters for tracking
    #         all_params = set(self.net_g.parameters())
            
    #         # Group 1: Custom LR params
    #         for key, cfg in custom_keys.items():
    #             group_params = []
    #             params_to_remove = set()
                
    #             # Find all parameters that match the key (e.g., 'backbone.fusion_module.layer1')
    #             for name, param in self.net_g.named_parameters():
    #                 if param in all_params and key in name:
    #                     group_params.append(param)
    #                     params_to_remove.add(param)
                
    #             if group_params:
    #                 # Apply the LR multiplier
    #                 current_lr = base_lr * cfg['lr_mult']
                    
    #                 # Also check for other custom settings like weight_decay
    #                 group_cfg = cfg.copy()
    #                 group_cfg['params'] = group_params
    #                 if 'lr' not in group_cfg:
    #                     group_cfg['lr'] = current_lr
                    
    #                 param_groups.append(group_cfg)
                
    #             # Remove these params from the main set so they don't get default LR
    #             all_params.difference_update(params_to_remove)

    #         # Group 2: All remaining params get the default LR
    #         param_groups.append({'params': list(all_params)})

    #     # 3. Create the optimizer with the new param_groups
    #     # optim_g_opt now only contains valid AdamW args (lr, betas, etc.)
    #     self.optimizer_g = torch.optim.AdamW(param_groups, **optim_g_opt)
    #     self.optimizers.append(self.optimizer_g)


    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    # def optimize_parameters(self, current_iter):
    #     self.optimizer_g.zero_grad()

    #     with autocast(enabled=self.use_amp):
    #         preds = self.net_g(self.lq)
    #         if not isinstance(preds, list):
    #             preds = [preds]

    #         self.output = preds[-1]

    #         loss_dict = OrderedDict()
    #         # pixel loss
    #         l_pix = 0.
    #         for pred in preds:
    #             l_pix += self.cri_pix(pred, self.gt) #此处统计batch的loss

    #         loss_dict['l_pix'] = l_pix

    #     self.amp_scaler.scale(l_pix).backward()
    #     self.amp_scaler.unscale_(self.optimizer_g) # 在梯度裁剪前先unscale梯度
    #     # l_pix.backward()

    #     if self.opt['train']['use_grad_clip']:
    #         torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
    #     # self.optimizer_g.step()
    #     self.amp_scaler.step(self.optimizer_g)
    #     self.amp_scaler.update()

    #     self.log_dict = self.reduce_loss_dict(loss_dict)

    #     if self.ema_decay > 0:
    #         self.model_ema(decay=self.ema_decay)

    def optimize_parameters(self, current_iter):
        # -------------------------------------------------------
        # 1. Setup Accumulation
        # -------------------------------------------------------
        # Get accumulation steps from config, default to 1 (no accumulation)
        # accum_steps = self.opt['train'].get('accum_steps', 1)
        accum_steps = 4

        # -------------------------------------------------------
        # 2. Forward Pass
        # -------------------------------------------------------
        with autocast(enabled=self.use_amp):
            preds = self.net_g(self.lq)
            if not isinstance(preds, list):
                preds = [preds]

            self.output = preds[-1]

            loss_dict = OrderedDict()
            # pixel loss
            l_pix = 0.
            for pred in preds:
                l_pix += self.cri_pix(pred, self.gt)

            # Keep original value for logging (so curves look normal)
            loss_dict['l_pix'] = l_pix.detach() 
            
            # CRITICAL: Scale loss down by accum_steps
            # Gradients sum up in PyTorch, so we must divide the loss
            l_pix_scaled = l_pix / accum_steps

        # -------------------------------------------------------
        # 3. Backward (Accumulate Gradients)
        # -------------------------------------------------------
        # We backward the SCALED loss
        self.amp_scaler.scale(l_pix_scaled).backward()

        # -------------------------------------------------------
        # 4. Optimizer Step (Conditional)
        # -------------------------------------------------------
        # Only update weights every 'accum_steps' iterations
        if (current_iter + 1) % accum_steps == 0:
            
            # Unscale gradients before clipping
            self.amp_scaler.unscale_(self.optimizer_g)

            # Gradient Clipping
            if self.opt['train']['use_grad_clip']:
                # Check config for specific clip value, default to 1.0 or 0.01
                clip_norm = self.opt['train'].get('grad_clip_norm', 0.01)
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), clip_norm)

            # Step & Update Scaler
            self.amp_scaler.step(self.optimizer_g)
            self.amp_scaler.update()
            
            # CRITICAL: Zero gradients HERE, only after stepping
            self.optimizer_g.zero_grad()

            # Update EMA only when we step the optimizer
            if self.ema_decay > 0:
                self.model_ema(decay=self.ema_decay)

        # -------------------------------------------------------
        # 5. Logging
        # -------------------------------------------------------
        self.log_dict = self.reduce_loss_dict(loss_dict)

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        """
        Override parent method to support Gradient Accumulation.
        We only update the LR (scheduler and warmup) when the optimizer actually steps.
        """
        accum_steps = self.opt['train'].get('accum_steps', 4)

        # Only update if the optimizer stepped in this iteration
        if (current_iter + 1) % accum_steps == 0:
            
            # 1. Step the main Schedulers (Cosine, MultiStep, etc.)
            for scheduler in self.schedulers:
                scheduler.step()
                
            # 2. Handle BasicSR's Manual Linear Warmup
            # (Copied from BaseModel but guarded by accum_steps)
            if current_iter < warmup_iter:
                # get initial lr for each group
                init_lr_g_l = self._get_init_lr()
                # modify warming-up learning rates
                warm_up_lr_l = []
                for init_lr_g in init_lr_g_l:
                    warm_up_lr_l.append(
                        [v / warmup_iter * current_iter for v in init_lr_g])
                # set learning rate
                self._set_lr(warm_up_lr_l)

    def pad_test(self, window_size):
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h -
                                  mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img)
                # pred = pred[0] # deep supervision
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def save_tensor_as_image(self, tensor, save_dir, name_prefix):
        """Save a segmentation or label tensor as a colored PNG image.
        Handles:
        - prediction tensors with shape [1, C, H, W] (argmax applied)
        - ground truth tensors with shape [1, H, W] or [H, W] (values in [0, 4])
        """
        import os
        import torch
        import numpy as np
        import cv2

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{name_prefix}.png")

        tensor = tensor.detach().cpu()

        # Handle different shapes
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        if tensor.dim() == 3:
            if tensor.shape[0] > 1:
                tensor = torch.argmax(tensor, dim=0)
            else:
                tensor = tensor.squeeze(0)

        tensor = tensor.numpy().astype(np.uint8)  # [H, W]

        # Define class colors: 0=black, 1–4 distinct colors
        colors = {
            0: (0, 0, 0),         # background - black
            1: (255, 0, 0),       # class 1 - red
            2: (0, 255, 0),       # class 2 - green
            3: (0, 0, 255),       # class 3 - blue
            4: (255, 255, 0),     # class 4 - yellow
        }

        # Colorize
        h, w = tensor.shape
        color_img = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id, color in colors.items():
            mask = tensor == cls_id
            color_img[mask] = color

        # Save image
        cv2.imwrite(save_path, color_img[:, :, ::-1])  # RGB→BGR for OpenCV

    # PST900 and MFNET
    # def nondist_validation(self, dataloader, current_iter, tb_logger,
    #                     save_img, rgb2bgr, use_image):
    #     """
    #     Dataset-level validation with correct global mIoU computation.
    #     """
    #     self.net_g.eval()
    #     from sklearn.metrics import confusion_matrix
    #     from functools import partial
    #     import numpy as np
        
    #     dataset_name = dataloader.dataset.opt['name']
    #     with_metrics = self.opt['val'].get('metrics') is not None
    #     num_classes = self.opt['val'].get('num_classes', 5)
        
    #     # Initialize global confusion matrix
    #     global_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    #     cnt = 0
        
    #     # Color map for visualization
    #     color_map = np.array([
    #         [0, 0, 0],       # Class 0: Background
    #         [255, 0, 0],     # Class 1: Red
    #         [0, 255, 0],     # Class 2: Green
    #         [0, 0, 255],     # Class 3: Blue
    #         [255, 255, 0],   # Class 4: Yellow
    #     ], dtype=np.uint8)
        
    #     window_size = self.opt['val'].get('window_size', 0)
    #     test = partial(self.pad_test, window_size) if window_size else self.nonpad_test
        
    #     with torch.no_grad():  # Ensure no gradients during validation
    #         for idx, val_data in enumerate(dataloader):
    #             img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
    #             self.feed_data(val_data)
                
    #             # Run inference
    #             test()
    #             visuals = self.get_current_visuals()
                
    #             # ============================================
    #             # 1. Process predictions for metrics FIRST
    #             # ============================================
    #             if with_metrics and 'gt' in visuals:
    #                 pred_tensor = visuals['result']  # [1, C, H, W] or [C, H, W]
    #                 gt_tensor = visuals['gt']        # [1, 1, H, W] or [1, H, W]
                    
    #                 # Convert to [H, W] format
    #                 if pred_tensor.dim() == 4:
    #                     assert pred_tensor.size(0) == 1, "Batch size must be 1"
    #                     pred_labels = torch.argmax(pred_tensor[0], dim=0)  # [H, W]
    #                 elif pred_tensor.dim() == 3:
    #                     pred_labels = torch.argmax(pred_tensor, dim=0)  # [H, W]
    #                 else:
    #                     raise ValueError(f"Unexpected pred shape: {pred_tensor.shape}")
                    
    #                 if gt_tensor.dim() == 4:
    #                     gt_labels = gt_tensor[0, 0]  # [H, W]
    #                 elif gt_tensor.dim() == 3:
    #                     gt_labels = gt_tensor[0]  # [H, W]
    #                 else:
    #                     gt_labels = gt_tensor
                    
    #                 # Convert to numpy
    #                 pred_np = pred_labels.cpu().numpy().astype(np.int64)
    #                 gt_np = gt_labels.cpu().numpy().astype(np.int64)
                    
    #                 # Flatten and filter valid pixels
    #                 pred_flat = pred_np.flatten()
    #                 gt_flat = gt_np.flatten()
    #                 mask = (gt_flat >= 0) & (gt_flat < num_classes)
                    
    #                 if mask.sum() > 0:
    #                     pred_flat = pred_flat[mask]
    #                     gt_flat = gt_flat[mask]
    #                     # Update global confusion matrix
    #                     global_cm += confusion_matrix(
    #                         gt_flat, pred_flat, 
    #                         labels=list(range(num_classes))
    #                     )
    #                 else:
    #                     print(f"Warning: No valid pixels in {img_name}")
                
    #             # ============================================
    #             # 2. Save visualizations
    #             # ============================================
    #             if save_img:
    #                 save_dir = osp.join(self.opt['path']['visualization'], dataset_name)
    #                 os.makedirs(save_dir, exist_ok=True)
                    
    #                 # Colorize segmentation masks
    #                 pred_tensor = visuals['result']
    #                 if pred_tensor.dim() == 4:
    #                     pred_labels = torch.argmax(pred_tensor[0], dim=0)
    #                 else:
    #                     pred_labels = torch.argmax(pred_tensor, dim=0)
                    
    #                 pred_colored = color_map[pred_labels.cpu().numpy()]
                    
    #                 save_img_path = osp.join(save_dir, f"{img_name}_pred.png")
    #                 imwrite(pred_colored, save_img_path)
                    
    #                 if 'gt' in visuals:
    #                     gt_tensor = visuals['gt']
    #                     if gt_tensor.dim() == 4:
    #                         gt_labels = gt_tensor[0, 0]
    #                     elif gt_tensor.dim() == 3:
    #                         gt_labels = gt_tensor[0]
    #                     else:
    #                         gt_labels = gt_tensor
                        
    #                     gt_colored = color_map[gt_labels.cpu().numpy()]
    #                     save_gt_img_path = osp.join(save_dir, f"{img_name}_gt.png")
    #                     imwrite(gt_colored, save_gt_img_path)
                
    #             # ============================================
    #             # 3. Clear GPU memory
    #             # ============================================
    #             del self.lq
    #             if hasattr(self, 'output'):
    #                 del self.output
    #             if hasattr(self, 'gt'):
    #                 del self.gt
    #             torch.cuda.empty_cache()
                
    #             cnt += 1
        
    #     # ============================================
    #     # 4. Compute dataset-level metrics
    #     # ============================================
    #     intersection = np.diag(global_cm)
    #     union = global_cm.sum(1) + global_cm.sum(0) - intersection
        
    #     # Only compute IoU for classes that exist
    #     valid_classes = union > 0
    #     iou_per_class = np.full(num_classes, np.nan)
    #     iou_per_class[valid_classes] = intersection[valid_classes] / union[valid_classes]
        
    #     mean_iou = np.nanmean(iou_per_class)
    #     pixel_acc = intersection.sum() / (global_cm.sum() + 1e-10)
        
    #     # ============================================
    #     # 5. Log results
    #     # ============================================
    #     print(f"\n{'='*50}")
    #     print(f"Validation Results: {dataset_name}")
    #     print(f"{'='*50}")
    #     print(f"Processed images: {cnt}")
    #     print(f"Mean IoU:         {mean_iou:.4f}")
    #     print(f"Pixel Accuracy:   {pixel_acc:.4f}")
    #     print(f"Valid classes:    {valid_classes.sum()} / {num_classes}")
    #     print(f"\nIoU per class:")
    #     for i in range(num_classes):
    #         if valid_classes[i]:
    #             print(f"  Class {i}: {iou_per_class[i]:.4f}")
    #         else:
    #             print(f"  Class {i}: N/A (not present)")
    #     print(f"{'='*50}\n")
        
    #     # TensorBoard logging
    #     if with_metrics and tb_logger is not None:
    #         tb_logger.add_scalar(f'{dataset_name}/mean_iou', mean_iou, current_iter)
    #         tb_logger.add_scalar(f'{dataset_name}/pixel_acc', pixel_acc, current_iter)
    #         for i in range(num_classes):
    #             if valid_classes[i]:
    #                 tb_logger.add_scalar(
    #                     f'{dataset_name}/class_{i}_iou', 
    #                     iou_per_class[i], 
    #                     current_iter
    #                 )
        
    #     self.net_g.train()
    #     return mean_iou
    # PST900 and MFNET

    # CART
    # import torch
    # import numpy as np
    # import os
    # import os.path as osp
    # from sklearn.metrics import confusion_matrix
    # from functools import partial
    # import cv2

    # # Note: This is designed to be a method inside your Model class.
    # # Ensure 'self.net_g', 'self.opt', 'self.feed_data', etc. are available.

    # def nondist_validation(self, dataloader, current_iter, tb_logger,
    #                     save_img, rgb2bgr, use_image):
    #     """
    #     CART Dataset-level validation with correct global mIoU computation.
    #     - Handles CART label remapping (IDs 2-11 -> 0-9).
    #     - Ignores IDs 0 (Unknown) and 1 (Background).
    #     """
    #     self.net_g.eval()
    #     import numpy as np
    #     import os
    #     import os.path as osp
    #     from sklearn.metrics import confusion_matrix
    #     from functools import partial
    #     import cv2
        
    #     dataset_name = dataloader.dataset.opt['name']
    #     with_metrics = self.opt['val'].get('metrics') is not None
    #     # Force 10 classes for CART (indices 0-9)
    #     num_classes = 10 
        
    #     # Initialize global confusion matrix
    #     global_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    #     cnt = 0
        
    #     # -------------------------------------------------------------------------
    #     # CART Color Map (RGB)
    #     # Maps indices 0-9 (which correspond to raw IDs 2-11) to colors
    #     # -------------------------------------------------------------------------
    #     # 0: Bare Ground (F2D8C4), 1: Rocky (594636), 2: Structure (A6A6A6), 3: Road (52595A)
    #     # 4: Shrubs (9BE600), 5: Trees (008A35), 6: Sky (00D8F5), 7: Water (0D7FFC)
    #     # 8: Vehicles (FFF900), 9: Person (FE00AA)
    #     color_map = np.array([
    #         [242, 216, 196], # 0: Bare Ground
    #         [89, 70, 54],    # 1: Rocky
    #         [166, 166, 166], # 2: Structure
    #         [82, 89, 90],    # 3: Road
    #         [155, 230, 0],   # 4: Shrubs
    #         [0, 138, 53],    # 5: Trees
    #         [0, 216, 245],   # 6: Sky
    #         [13, 127, 252],  # 7: Water
    #         [255, 249, 0],   # 8: Vehicles
    #         [254, 0, 170],   # 9: Person
    #     ], dtype=np.uint8)
        
    #     if rgb2bgr:
    #         # Flip RGB to BGR for OpenCV saving
    #         color_map = color_map[:, ::-1]

    #     window_size = self.opt['val'].get('window_size', 0)
    #     test = partial(self.pad_test, window_size) if window_size else self.nonpad_test
        
    #     with torch.no_grad():  # Ensure no gradients during validation
    #         for idx, val_data in enumerate(dataloader):
    #             img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
    #             self.feed_data(val_data)
                
    #             # Run inference
    #             test()
    #             visuals = self.get_current_visuals()
                
    #             # ============================================
    #             # 1. Process predictions for metrics FIRST
    #             # ============================================
    #             if with_metrics and 'gt' in visuals:
    #                 pred_tensor = visuals['result']  # [1, C, H, W]
    #                 gt_tensor = visuals['gt']        # [1, 1, H, W] or [1, H, W]
                    
    #                 # --- Get Prediction Labels (0-9) ---
    #                 if pred_tensor.dim() == 4:
    #                     pred_labels = torch.argmax(pred_tensor[0], dim=0) # [H, W]
    #                 else:
    #                     pred_labels = torch.argmax(pred_tensor, dim=0)
                    
    #                 # --- Get GT Labels (Raw 0-11) ---
    #                 if gt_tensor.dim() == 4:
    #                     gt_labels = gt_tensor[0, 0] # [H, W]
    #                 elif gt_tensor.dim() == 3:
    #                     gt_labels = gt_tensor[0]
    #                 else:
    #                     gt_labels = gt_tensor
                    
    #                 pred_np = pred_labels.cpu().numpy().astype(np.int64)
    #                 gt_np = gt_labels.cpu().numpy().astype(np.int64)
                    
    #                 # --- CART SPECIFIC: Remap & Filter GT ---
    #                 # 1. Identify valid pixels: Raw IDs 2 through 11
    #                 valid_mask = (gt_np >= 2) & (gt_np <= 11)
                    
    #                 # 2. Remap valid pixels: 2->0, 3->1, ... 11->9
    #                 gt_remapped = gt_np.copy()
    #                 gt_remapped[valid_mask] = gt_remapped[valid_mask] - 2
    #                 gt_remapped[~valid_mask] = -1 # Mark invalid as -1
                    
    #                 # Flatten arrays
    #                 pred_flat = pred_np.flatten()
    #                 gt_flat = gt_remapped.flatten()
                    
    #                 # 3. Filter for confusion matrix
    #                 # Keep only pixels where GT is 0-9 (derived from valid_mask)
    #                 mask = (gt_flat >= 0) & (gt_flat < num_classes)
                    
    #                 if mask.sum() > 0:
    #                     pred_flat = pred_flat[mask]
    #                     gt_flat = gt_flat[mask]
                        
    #                     # Update global confusion matrix
    #                     global_cm += confusion_matrix(
    #                         gt_flat, pred_flat, 
    #                         labels=list(range(num_classes))
    #                     )
    #                 else:
    #                     # It's possible an image contains ONLY background/unknown
    #                     pass 
                
    #             # ============================================
    #             # 2. Save visualizations
    #             # ============================================
    #             if save_img:
    #                 save_dir = osp.join(self.opt['path']['visualization'], dataset_name)
    #                 os.makedirs(save_dir, exist_ok=True)
                    
    #                 # --- Colorize Prediction ---
    #                 # pred_labels is already 0-9, maps directly to color_map
    #                 pred_colored = color_map[pred_labels.cpu().numpy()]
    #                 save_img_path = osp.join(save_dir, f"{img_name}_pred.png")
    #                 cv2.imwrite(save_img_path, pred_colored)
                    
    #                 # --- Colorize Ground Truth ---
    #                 if 'gt' in visuals:
    #                     gt_vis = gt_labels.cpu().numpy().copy() # Raw 0-11
    #                     h, w = gt_vis.shape
                        
    #                     # Initialize black image (for IDs 0 and 1)
    #                     gt_colored = np.zeros((h, w, 3), dtype=np.uint8)
                        
    #                     # Mask valid pixels (2-11)
    #                     valid_mask_2d = (gt_vis >= 2) & (gt_vis <= 11)
                        
    #                     # Remap indices for color lookup (2->0, etc)
    #                     valid_indices = gt_vis[valid_mask_2d] - 2
                        
    #                     # Assign colors
    #                     gt_colored[valid_mask_2d] = color_map[valid_indices]
                        
    #                     save_gt_img_path = osp.join(save_dir, f"{img_name}_gt.png")
    #                     cv2.imwrite(save_gt_img_path, gt_colored)
                
    #             # ============================================
    #             # 3. Clear GPU memory
    #             # ============================================
    #             del self.lq
    #             if hasattr(self, 'output'):
    #                 del self.output
    #             if hasattr(self, 'gt'):
    #                 del self.gt
    #             torch.cuda.empty_cache()
                
    #             cnt += 1
        
    #     # ============================================
    #     # 4. Compute dataset-level metrics
    #     # ============================================
    #     intersection = np.diag(global_cm)
    #     union = global_cm.sum(1) + global_cm.sum(0) - intersection
        
    #     # Only compute IoU for classes that exist in the union
    #     valid_classes = union > 0
    #     iou_per_class = np.full(num_classes, np.nan)
    #     iou_per_class[valid_classes] = intersection[valid_classes] / union[valid_classes]
        
    #     mean_iou = np.nanmean(iou_per_class)
    #     pixel_acc = intersection.sum() / (global_cm.sum() + 1e-10)
        
    #     # ============================================
    #     # 5. Log results
    #     # ============================================
    #     print(f"\n{'='*50}")
    #     print(f"Validation Results: {dataset_name}")
    #     print(f"{'='*50}")
    #     print(f"Processed images: {cnt}")
    #     print(f"Mean IoU:         {mean_iou:.4f}")
    #     print(f"Pixel Accuracy:   {pixel_acc:.4f}")
        
    #     class_names = [
    #         "Bare Ground", "Rocky", "Structure", "Road", "Shrubs", 
    #         "Trees", "Sky", "Water", "Vehicles", "Person"
    #     ]
        
    #     print(f"\nIoU per class:")
    #     for i in range(num_classes):
    #         name = class_names[i] if i < len(class_names) else f"Class {i}"
    #         if valid_classes[i]:
    #             print(f"  {name:<12}: {iou_per_class[i]:.4f}")
    #         else:
    #             print(f"  {name:<12}: N/A (not present)")
    #     print(f"{'='*50}\n")
        
    #     # TensorBoard logging
    #     if with_metrics and tb_logger is not None:
    #         tb_logger.add_scalar(f'{dataset_name}/mean_iou', mean_iou, current_iter)
    #         tb_logger.add_scalar(f'{dataset_name}/pixel_acc', pixel_acc, current_iter)
    #         for i in range(num_classes):
    #             if valid_classes[i]:
    #                 name = class_names[i] if i < len(class_names) else f"class_{i}"
    #                 tb_logger.add_scalar(
    #                     f'{dataset_name}/{name}_iou', 
    #                     iou_per_class[i], 
    #                     current_iter
    #                 )
        
    #     self.net_g.train()
    #     return mean_iou
    
    # CART



    # ENH Version
    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        # self.net_g.eval()
        # # for name, module in self.net_g.named_modules():
        # #     print(name, module.training)
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:

                if self.opt['is_train']:

                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')

                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                img_name,
                                                f'{img_name}_{current_iter}_gt.png')
                else:

                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_gt.png')

                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        # # print("metric module:", metric_module)
                        # # print("visuals shape and tyoe", visuals['result'].shape, visuals['gt'].shape, type(visuals['result']))
                        # # --- Save raw tensors for debugging ---
                        # base_save_dir = "validation_outputs"  # fixed independent path
                        # save_dir = os.path.join(base_save_dir, f"iter_{current_iter}")
                        # os.makedirs(save_dir, exist_ok=True)

                        # # self.save_tensor_as_image(visuals['result'], save_dir, f"{img_name}_pred")
                        # # self.save_tensor_as_image(visuals['gt'], save_dir, f"{img_name}_gt")

                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        # self.net_g.train()
        # for name, module in self.net_g.fusion_module.named_modules():
        #     print(name, module.training)
        return current_metric

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter, **kwargs):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter, **kwargs)

    def save_best(self, best_metric, param_key='params'):
        psnr = best_metric['psnr']
        cur_iter = best_metric['iter']
        save_filename = f'best_psnr_{psnr:.2f}_{cur_iter}.pth'
        exp_root = self.opt['path']['experiments_root']
        save_path = os.path.join(
            self.opt['path']['experiments_root'], save_filename)

        if not os.path.exists(save_path):
            for r_file in glob.glob(f'{exp_root}/best_*'):
                os.remove(r_file)
            net = self.net_g

            net = net if isinstance(net, list) else [net]
            param_key = param_key if isinstance(
                param_key, list) else [param_key]
            assert len(net) == len(
                param_key), 'The lengths of net and param_key should be the same.'

            save_dict = {}
            for net_, param_key_ in zip(net, param_key):
                net_ = self.get_bare_model(net_)
                state_dict = net_.state_dict()
                for key, param in state_dict.items():
                    if key.startswith('module.'):  # remove unnecessary 'module.'
                        key = key[7:]
                    state_dict[key] = param.cpu()
                save_dict[param_key_] = state_dict

            torch.save(save_dict, save_path)
