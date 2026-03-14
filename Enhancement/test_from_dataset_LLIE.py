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

import utils 
from basicsr.models import create_model
from basicsr.utils.options import parse
from basicsr.data.paired_image_dataset import Dataset_PairedImage_Enhancement

def self_ensemble(x, x_second, model):
    """Self-ensemble using flips and rotations on combined inputs."""
    x_combined = torch.cat([x, x_second], dim=1)
    
    def forward_combined_transformed(x_c, hflip, vflip, rotate, model):
        if hflip: x_c = torch.flip(x_c, (-2,))
        if vflip: x_c = torch.flip(x_c, (-1,))
        if rotate: x_c = torch.rot90(x_c, dims=(-2, -1))
        
        out = model(x_c)
        
        if rotate: out = torch.rot90(out, dims=(-2, -1), k=3)
        if vflip: out = torch.flip(out, (-1,))
        if hflip: out = torch.flip(out, (-2,))
        return out

    t = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            for rot in [False, True]:
                t.append(forward_combined_transformed(x_combined, hflip, vflip, rot, model))
    
    return torch.mean(torch.stack(t), dim=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dual-Input Image Enhancement PSNR/SSIM Test')
    parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
    parser.add_argument('--output_dir', default='', type=str, help='Directory for output')
    parser.add_argument('--opt', type=str, default='', help='Path to option YAML file.')
    parser.add_argument('--weights', default='', type=str, help='Path to weights')
    parser.add_argument('--dataset', default="", type=str, help='Test Dataset Name') 
    parser.add_argument('--gpus', type=str, default="0", help='GPU devices.')
    parser.add_argument('--self_ensemble', action='store_true', help='Use self-ensemble to obtain better results')

    args = parser.parse_args()

    gpu_list = ','.join(str(x) for x in args.gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print(f"export CUDA_VISIBLE_DEVICES={gpu_list}")

    print(f"Dataset: {args.dataset}")
    opt = parse(args.opt, is_train=False)
    opt['dist'] = False

    model_restoration = create_model(opt).net_g
    checkpoint = torch.load(args.weights)

    try:
        model_restoration.load_state_dict(checkpoint['params'])
    except KeyError:
        # Fallback for weights saved with DataParallel 'module.' prefix
        new_checkpoint = {'module.' + k: v for k, v in checkpoint['params'].items()}
        model_restoration.load_state_dict(new_checkpoint)

    print(f"===> Testing using weights: {args.weights}")
    model_restoration.cuda()
    model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()

    factor = 4 
    dataset_name = args.dataset
    config_name = os.path.basename(args.opt).split('.')[0]
    checkpoint_name = os.path.basename(args.weights).split('.')[0]
    
    result_dir = os.path.join(args.result_dir, dataset_name, config_name, checkpoint_name)
    result_dir_input = os.path.join(args.result_dir, dataset_name, 'input')
    result_dir_gt = os.path.join(args.result_dir, dataset_name, 'gt')

    os.makedirs(result_dir, exist_ok=True)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
    os.makedirs(result_dir_input, exist_ok=True)
    os.makedirs(result_dir_gt, exist_ok=True)

    dataset_opt = opt['datasets']['val']
    dataset_opt['phase'] = 'test'
    
    # Handle home directory expansion for dataset paths
    for key in ['dataroot_gt', 'dataroot_lq']:
        if '~' in dataset_opt.get(key, ''):
            dataset_opt[key] = os.path.expanduser('~') + dataset_opt[key][1:]
            
    dataset = Dataset_PairedImage_Enhancement(dataset_opt)
    print(f"Test dataset length: {len(dataset)}")
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    psnr, ssim = [], []

    with torch.inference_mode():
        for data_batch in tqdm(dataloader):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            # Input shape expected: [B, 4, H, W]
            combined_input = data_batch['lq'].cuda() 
            target = data_batch['gt'].cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
            inp_path = data_batch['lq_path'][0]

            # Split the 4-channel input: RGB (first 3) and Thermal (4th)
            input_ = combined_input[:, :3, :, :]
            input_second = combined_input[:, 3:, :, :] 
            input_save = input_.cpu().permute(0, 2, 3, 1).squeeze(0).numpy()

            h, w = input_.shape[2], input_.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
            input_second = F.pad(input_second, (0, padw, 0, padh), 'reflect')

            if args.self_ensemble:
                restored = self_ensemble(torch.cat([input_, input_second], dim=1), model_restoration)
            else:
                restored = model_restoration(torch.cat([input_, input_second], dim=1))

            restored = restored[:, :, :h, :w]
            restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            psnr.append(utils.PSNR(target, restored))
            ssim.append(utils.calculate_ssim(img_as_ubyte(target), img_as_ubyte(restored)))

            type_id = os.path.basename(os.path.dirname(inp_path))
            save_result_dir = os.path.join(result_dir, type_id)
            save_input_dir = os.path.join(result_dir_input, type_id)
            save_gt_dir = os.path.join(result_dir_gt, type_id)
            
            os.makedirs(save_result_dir, exist_ok=True)
            os.makedirs(save_input_dir, exist_ok=True)
            os.makedirs(save_gt_dir, exist_ok=True)

            filename = os.path.splitext(os.path.basename(inp_path))[0] + '.png'

            utils.save_img(os.path.join(save_result_dir, filename), img_as_ubyte(restored))
            utils.save_img(os.path.join(save_input_dir, filename), img_as_ubyte(input_save))
            utils.save_img(os.path.join(save_gt_dir, filename), img_as_ubyte(target))

    print("\n============== RESULTS ==============")
    print(f"PSNR: {np.mean(psnr):.6f}")
    print(f"SSIM: {np.mean(ssim):.6f}")
    print("=====================================")