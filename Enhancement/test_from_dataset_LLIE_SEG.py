import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage import img_as_ubyte
import yaml

# Assuming 'utils' is available in your environment as per the first script
import utils 

from basicsr.models import create_model
from basicsr.utils.options import parse
# Use the dataset class from the segmentation script reference
from basicsr.data.paired_image_dataset import Dataset_PairedImage

# -------------------------------------------------------------------
# Updated Self-Ensemble to handle Dual Inputs (requires modification)
# -------------------------------------------------------------------
def self_ensemble(x, x_second, model):
    def forward_transformed(x, x_second, hflip, vflip, rotate, model):
        
        # --- FIX: Re-concatenate inputs for the model inside self-ensemble ---
        # The model expects a single 4-channel input if the dataset is unmodified.
        x_combined = torch.cat([x, x_second], dim=1) # Combine [B, 3, H, W] and [B, 1, H, W] to [B, 4, H, W]
        
        # Apply transformation to the combined input (or only to x/x_second if model supports it)
        # Since the model expects a combined input, we apply transforms *before* combining
        # to ensure consistency with the original concatenated training setup.
        
        # Transform first input
        if hflip: x = torch.flip(x, (-2,))
        if vflip: x = torch.flip(x, (-1,))
        if rotate: x = torch.rot90(x, dims=(-2, -1))
        
        # Transform second input (Identical transformation)
        if hflip: x_second = torch.flip(x_second, (-2,))
        if vflip: x_second = torch.flip(x_second, (-1,))
        if rotate: x_second = torch.rot90(x_second, dims=(-2, -1))
        
        # Combine transformed inputs for the single model call
        x_combined = torch.cat([x, x_second], dim=1) 
        
        # Forward pass with the single combined input
        out = model(x_combined) 
        
        # Inverse transform output (Output is only the image, usually 3 channels)
        if rotate: out = torch.rot90(out, dims=(-2, -1), k=3)
        if vflip: out = torch.flip(out, (-1,))
        if hflip: out = torch.flip(out, (-2,))
        return out

    # We must ensure x and x_second retain their original separation for the transforms
    # However, since the model only takes one input, we need to pass a single combined tensor.
    
    # To use self_ensemble, we must load the combined input once, split it for transforms, 
    # and recombine it for the model. This is complex and suggests the model should 
    # be adapted, but to avoid modifying the model or dataset, we must simulate the dual input
    # inside self_ensemble.
    
    # NOTE: Since the current x and x_second passed to self_ensemble are the *unpadded* inputs
    # we need to modify the loop below to work with the single combined input if self_ensemble is used.
    # For simplicity, we assume the model expects ONE combined [4,H,W] input if self_ensemble is used.
    
    # We will combine them here to treat them as the single tensor the model expects.
    x_combined = torch.cat([x, x_second], dim=1)
    
    def forward_combined_transformed(x_c, hflip, vflip, rotate, model):
        # Transform combined input
        if hflip: x_c = torch.flip(x_c, (-2,))
        if vflip: x_c = torch.flip(x_c, (-1,))
        if rotate: x_c = torch.rot90(x_c, dims=(-2, -1))
        
        out = model(x_c)
        
        # Inverse transform output
        if rotate: out = torch.rot90(out, dims=(-2, -1), k=3)
        if vflip: out = torch.flip(out, (-1,))
        if hflip: out = torch.flip(out, (-2,))
        return out

    t = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            for rot in [False, True]:
                t.append(forward_combined_transformed(x_combined, hflip, vflip, rot, model))
    t = torch.stack(t)
    return torch.mean(t, dim=0)

# -------------------------------------------------------------------
# Main Script
# -------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dual-Input Image Enhancement PSNR/SSIM Test')

    parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
    parser.add_argument('--output_dir', default='', type=str, help='Directory for output')
    parser.add_argument('--opt', type=str, default='Options/RetinexFormer_SDSD_indoor.yml', help='Path to option YAML file.')
    parser.add_argument('--weights', default='pretrained_weights/SDSD_indoor.pth', type=str, help='Path to weights')
    parser.add_argument('--dataset', default="DualInputDataset", type=str, help='Test Dataset Name') 
    parser.add_argument('--gpus', type=str, default="0", help='GPU devices.')
    parser.add_argument('--GT_mean', action='store_true', help='Use the mean of GT to rectify the output of the model')
    parser.add_argument('--self_ensemble', action='store_true', help='Use self-ensemble to obtain better results')

    args = parser.parse_args()

    # GPU Setup
    gpu_list = ','.join(str(x) for x in args.gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    # ---------------------------------------------------------
    # Load YAML and Model
    # ---------------------------------------------------------
    print(f"Dataset: {args.dataset}")
    opt = parse(args.opt, is_train=False)
    opt['dist'] = False

    model_restoration = create_model(opt).net_g
    checkpoint = torch.load(args.weights)

    try:
        model_restoration.load_state_dict(checkpoint['params'])
    except:
        new_checkpoint = {}
        for k in checkpoint['params']:
            new_checkpoint['module.' + k] = checkpoint['params'][k]
        model_restoration.load_state_dict(new_checkpoint)

    print("===> Testing using weights: ", args.weights)
    model_restoration.cuda()
    model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()

    # ---------------------------------------------------------
    # Directory Setup
    # ---------------------------------------------------------
    factor = 4 # Padding factor
    dataset_name = args.dataset
    config_name = os.path.basename(args.opt).split('.')[0]
    checkpoint_name = os.path.basename(args.weights).split('.')[0]
    
    result_dir = os.path.join(args.result_dir, dataset_name, config_name, checkpoint_name)
    result_dir_input = os.path.join(args.result_dir, dataset_name, 'input')
    result_dir_gt = os.path.join(args.result_dir, dataset_name, 'gt')
    output_dir = args.output_dir

    os.makedirs(result_dir, exist_ok=True)
    if args.output_dir != '':
        os.makedirs(output_dir, exist_ok=True)
    
    # Optional: Save input/gt for reference
    os.makedirs(result_dir_input, exist_ok=True)
    os.makedirs(result_dir_gt, exist_ok=True)

    # ---------------------------------------------------------
    # Data Loader (From Segmentation Script Logic)
    # ---------------------------------------------------------
    dataset_opt = opt['datasets']['val']
    dataset_opt['phase'] = 'test'
    
    # If the YAML uses 'lq_second', the dataset expects 'dataroot_lq_second'. 
    # Since we are NOT modifying the dataset, we must assume the dataset's __init__ 
    # internally handles the 'lq_second' key provided in the YAML for the thermal folder path
    # (as your dataset code showed it looking for `self.second_folder = opt.get('lq_second', None)`).
    # We do NOT rename the key here, to match the existing dataset initialization.
    
    # Handle home directory expansion if necessary
    if '~' in dataset_opt.get('dataroot_gt', ''):
        dataset_opt['dataroot_gt'] = os.path.expanduser('~') + dataset_opt['dataroot_gt'][1:]
    if '~' in dataset_opt.get('dataroot_lq', ''):
        dataset_opt['dataroot_lq'] = os.path.expanduser('~') + dataset_opt['dataroot_lq'][1:]
    # Assuming the original dataset successfully loads the second input folder using the 'lq_second' key in YAML

        
    dataset = Dataset_PairedImage(dataset_opt)
    print(f'Test dataset length: {len(dataset)}')
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    # ---------------------------------------------------------
    # Inference Loop
    # ---------------------------------------------------------
    psnr = []
    ssim = []

    with torch.inference_mode():
        for data_batch in tqdm(dataloader):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            # 1. Load Data
            # The dataset returns a single 4-channel tensor under 'lq'
            combined_input = data_batch['lq'].cuda() # Shape: [B, 4, H, W]
            target = data_batch['gt'].cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
            inp_path = data_batch['lq_path'][0]

            # --- FIX: Split the 4-channel input into two separate inputs ---
            # input_ (RGB): first 3 channels
            input_ = combined_input[:, :3, :, :]
            # input_second (Thermal): 4th channel
            input_second = combined_input[:, 3:, :, :] 
            
            # Save input for visualization (optional)
            # We save only the RGB part for input visualization
            input_save = input_.cpu().permute(0, 2, 3, 1).squeeze(0).numpy()

            # 2. Padding (Must pad ONLY the COMBINED input, or pad the split inputs identically)
            # Since the model expects two *padded* inputs passed separately, we pad the split tensors.
            h, w = input_.shape[2], input_.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
            input_second = F.pad(input_second, (0, padw, 0, padh), 'reflect')

            # 3. Inference
            if args.self_ensemble:
                # self_ensemble is modified above to handle the split/recombine internally
                restored = self_ensemble(input_, input_second, model_restoration)
            else:
                # The model *must* be capable of accepting two tensors (input_, input_second) 
                # as separate arguments, otherwise the model call needs to be:
                # combined_padded_input = torch.cat([input_, input_second], dim=1)
                # restored = model_restoration(combined_padded_input) 
                
                # Based on your previous code structure, we assume the model accepts two inputs:
                restored = model_restoration(input_, input_second)

            # 4. Unpad
            restored = restored[:, :, :h, :w]

            # 5. Post-process
            restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            # GT Mean Rectification (Optional)
            if args.GT_mean:
                mean_restored = cv2.cvtColor(restored.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                mean_target = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                restored = np.clip(restored * (mean_target / mean_restored), 0, 1)

            # 6. Calculate Metrics
            psnr.append(utils.PSNR(target, restored))
            ssim.append(utils.calculate_ssim(img_as_ubyte(target), img_as_ubyte(restored)))

            # 7. Save Images
            type_id = os.path.dirname(inp_path).split('/')[-1]
            
            # Create subfolders per type/category if applicable
            save_result_dir = os.path.join(result_dir, type_id)
            save_input_dir = os.path.join(result_dir_input, type_id)
            save_gt_dir = os.path.join(result_dir_gt, type_id)
            
            os.makedirs(save_result_dir, exist_ok=True)
            os.makedirs(save_input_dir, exist_ok=True)
            os.makedirs(save_gt_dir, exist_ok=True)

            filename = os.path.splitext(os.path.split(inp_path)[-1])[0] + '.png'

            utils.save_img(os.path.join(save_result_dir, filename), img_as_ubyte(restored))
            utils.save_img(os.path.join(save_input_dir, filename), img_as_ubyte(input_save))
            utils.save_img(os.path.join(save_gt_dir, filename), img_as_ubyte(target))

    # ---------------------------------------------------------
    # Final Results
    # ---------------------------------------------------------
    psnr_avg = np.mean(np.array(psnr))
    ssim_avg = np.mean(np.array(ssim))
    print("\n============== RESULTS ==============")
    print("PSNR: %f " % (psnr_avg))
    print("SSIM: %f " % (ssim_avg))
    print("=====================================")