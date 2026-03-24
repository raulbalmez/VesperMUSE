"""
Multi-Modal Segmentation Testing Script
Inputs: RGB + Thermal -> semantic logits -> mIoU & pixel accuracy
Dynamically handles CART or PST900 dataset classes.
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from skimage.color import label2rgb
from sklearn.metrics import confusion_matrix
import cv2

from basicsr.models import create_model
from basicsr.utils.options import parse
from basicsr.data.paired_image_dataset import Dataset_PairedImage_Segmentation


def update_confusion_matrix(cm, pred, gt, num_classes, dataset):
    """Updates the global confusion matrix, handling dataset-specific remapping."""
    pred = pred.flatten()
    gt = gt.flatten()

    if dataset.lower() == "cart":
        # Ignore IDs 0 and 1, remap 2-11 to 0-9
        valid_mask = (gt >= 2) & (gt <= 11)
        pred = pred[valid_mask]
        gt = gt[valid_mask]
        gt = gt - 2
        
        # Clean up out-of-bounds predictions
        mask_safe = (pred >= 0) & (pred < num_classes) & (gt >= 0) & (gt < num_classes)
        pred = pred[mask_safe]
        gt = gt[mask_safe]

    elif dataset.lower() == "pst900":
        mask = (gt >= 0) & (gt < num_classes)
        gt = gt[mask]
        pred = pred[mask]
        
    else:
        raise ValueError(f"Unknown dataset option: {dataset}")

    cm += confusion_matrix(gt, pred, labels=list(range(num_classes)))
    return cm


def compute_global_metrics(cm):
    intersection = np.diag(cm)
    union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - intersection
    
    iou = intersection / np.maximum(union, 1e-10)
    miou = np.mean(iou)

    pixel_acc = intersection.sum() / (cm.sum() + 1e-10)
    m_acc = np.mean(pixel_acc)
    
    return miou, m_acc, pixel_acc, iou


def forward_tta(x, model, scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75]):
    """Applies multi-scale and horizontal flip test-time augmentation."""
    b, c, h, w = x.shape
    logits_sum = 0
    cnt = 0

    for scale in scales:
        if scale != 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            x_s = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
        else:
            x_s = x
            new_h, new_w = h, w

        factor = 32
        H_pad = (new_h + factor - 1) // factor * factor
        W_pad = (new_w + factor - 1) // factor * factor
        pad_h = H_pad - new_h
        pad_w = W_pad - new_w
        
        x_s_pad = F.pad(x_s, (0, pad_w, 0, pad_h), mode='reflect')

        for flip in [False, True]:
            input_tensor = torch.flip(x_s_pad, dims=[-1]) if flip else x_s_pad
            out = model(input_tensor)
            
            if isinstance(out, (tuple, list)):
                out = out[0]

            if flip:
                out = torch.flip(out, dims=[-1])

            out = out[..., :new_h, :new_w]

            if scale != 1.0:
                out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)

            logits_sum += out
            cnt += 1

    return logits_sum / cnt


def main(args):
    opt = parse(args.opt, is_train=False)
    opt['dist'] = False

    model = create_model(opt).net_g
    checkpoint = torch.load(args.weights)
    raw_state_dict = checkpoint.get("params", checkpoint)
    
    # Strip "module." prefix from weights if present
    cleaned_state_dict = {
        (k[7:] if k.startswith('module.') else k): v 
        for k, v in raw_state_dict.items()
    }

    load_status = model.load_state_dict(cleaned_state_dict, strict=False)

    model = model.cuda()
    model.eval()
    model = torch.nn.DataParallel(model)

    print(f"\n✔ Loaded weights: {args.weights}")
    
    model_keys = set(model.module.state_dict().keys())
    ckpt_keys = set(cleaned_state_dict.keys())
    found_keys = list(model_keys.intersection(ckpt_keys))
    
    print(f"\n✅ FOUND PARAMS (Total: {len(found_keys)}):")
    for k in found_keys[:5]:
        print(f"  - {k}")
    if len(found_keys) > 5: print("  ... (and more)")

    print(f"\n❌ MISSING PARAMS (Total: {len(load_status.missing_keys)}):")
    for k in load_status.missing_keys[:5]:
        print(f"  - {k}")
    if len(load_status.missing_keys) > 5: print("  ... (and more)")

    print(f"\n➕ SUPPLEMENTARY PARAMS (Total: {len(load_status.unexpected_keys)}):")
    for k in load_status.unexpected_keys[:5]:
        print(f"  - {k}")
    if len(load_status.unexpected_keys) > 5: print("  ... (and more)")
    print("-" * 50)

    dataset_opt = opt["datasets"]["val"]
    dataset_opt["phase"] = "test"

    dataset = Dataset_PairedImage_Segmentation(dataset_opt)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"Dataset length: {len(dataset)}")
    print(f"Active Dataset Logic: {args.dataset.upper()}")

    num_classes = args.num_classes 
    factor = 32 

    os.makedirs(args.save_dir, exist_ok=True)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            torch.cuda.empty_cache()

            x = batch["lq"].cuda()
            gt = batch["gt"].squeeze(0).cpu().numpy().astype(np.int32)
            name = os.path.splitext(os.path.basename(batch["lq_path"][0]))[0]
            _, _, h, w = x.shape

            if args.tta:
                logits = forward_tta(x, model, scales=[0.9, 1.0, 1.1])
            else:
                H = (h + factor - 1) // factor * factor
                W = (w + factor - 1) // factor * factor
                pad_h = H - h
                pad_w = W - w
                x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
                
                logits = model(x_pad)
                
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                    
                logits = logits[:, :, :h, :w]

            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            cm = update_confusion_matrix(cm, pred, gt, num_classes, args.dataset)

            color_mask = label2rgb(pred, bg_label=0)
            color_mask = (color_mask * 255).astype(np.uint8)

            cv2.imwrite(
                os.path.join(args.save_dir, f"{name}.png"),
                cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
            )

    miou, m_acc, pixacc, iou_per_class = compute_global_metrics(cm)

    print("\n============== RESULTS ==============")
    if args.dataset.lower() == "cart":
        print(f"Evaluated on CART | {num_classes} classes (Valid IDs 2-11 mapped to 0-9)")
    else:
        print(f"Evaluated on PST900 | {num_classes} classes")
        
    print(f"Mean IoU:       {miou:.4f}")
    print(f"Pixel Accuracy: {pixacc:.4f}")
    print("IoU per class:")
    for i, score in enumerate(iou_per_class):
        print(f"  Class {i}: {score:.4f}")
    print("=====================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, required=True, help="YAML config")
    parser.add_argument("--weights", type=str, required=True, help="checkpoint .pth")
    parser.add_argument("--save_dir", type=str, default="./results_seg/")
    parser.add_argument("--dataset", type=str, required=True, choices=["cart", "pst900"], 
                        help="Dataset logic to apply to confusion matrix (cart or pst900)")
    parser.add_argument("--num_classes", type=int, required=True) 
    parser.add_argument("--tta", action="store_true")

    args = parser.parse_args()
    main(args)