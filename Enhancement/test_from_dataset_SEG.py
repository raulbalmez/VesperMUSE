# PST900 and MFNet eval start

# """
# Multi-Modal Segmentation Testing Script
# - Inputs: RGB + Thermal
# - Output: semantic logits
# - Evaluation: mIoU (global), pixel accuracy
# - Dataset: Paired RGB + Thermal + GT masks
# """

# import os
# import argparse
# import numpy as np
# from tqdm import tqdm
# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from skimage.color import label2rgb
# from sklearn.metrics import confusion_matrix
# import cv2

# from basicsr.models import create_model
# from basicsr.utils.options import parse
# from basicsr.data.paired_image_dataset import Dataset_PairedImage


# # ============================================================
# # Utility: Update confusion matrix
# # ============================================================
# def update_confusion_matrix(cm, pred, gt, num_classes):
#     """
#     pred: H×W, int
#     gt:   H×W, int
#     """
#     pred = pred.flatten()
#     gt = gt.flatten()

#     # keep only valid gt
#     mask = (gt >= 0) & (gt < num_classes)
#     gt = gt[mask]
#     pred = pred[mask]

#     cm += confusion_matrix(gt, pred, labels=list(range(num_classes)))
#     return cm


# # ============================================================
# # Compute global metrics after processing entire dataset
# # ============================================================
# def compute_global_metrics(cm):
#     intersection = np.diag(cm)
#     union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - intersection
#     iou = intersection / np.maximum(union, 1e-10)
#     miou = np.mean(iou)

#     pixel_acc = intersection.sum() / (cm.sum() + 1e-10)
#     m_acc = np.mean(pixel_acc)
#     return miou,m_acc, pixel_acc, iou


# # ============================================================
# # Optional: Test-time augmentation (horizontal flip)
# # ============================================================
# def forward_tta(x, model, scales=[0.7, 1.1, 1.5]):
#     """
#     Multi-Scale + Flip Test Time Augmentation
#     1. Resize input to scale
#     2. Pad to multiple of 32
#     3. Predict (original + flip)
#     4. Unpad
#     5. Resize back to original
#     6. Average
#     """
#     b, c, h, w = x.shape
#     logits_sum = 0
#     cnt = 0

#     for scale in scales:
#         # -------------------------------------------
#         # 1. Resize Input
#         # -------------------------------------------
#         if scale != 1.0:
#             new_h, new_w = int(h * scale), int(w * scale)
#             x_s = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
#         else:
#             x_s = x
#             new_h, new_w = h, w

#         # -------------------------------------------
#         # 2. Pad to multiple of 32 (Model Requirement)
#         # -------------------------------------------
#         factor = 32
#         H_pad = (new_h + factor - 1) // factor * factor
#         W_pad = (new_w + factor - 1) // factor * factor
#         pad_h = H_pad - new_h
#         pad_w = W_pad - new_w
        
#         # Use reflect padding to avoid hard edges
#         x_s_pad = F.pad(x_s, (0, pad_w, 0, pad_h), mode='reflect')

#         # -------------------------------------------
#         # 3. Loop Flips
#         # -------------------------------------------
#         for flip in [False, True]:
#             # Flip input
#             input_tensor = torch.flip(x_s_pad, dims=[-1]) if flip else x_s_pad

#             # Forward
#             out = model(input_tensor)
            
#             # Handle if model returns tuple (logits, aux)
#             if isinstance(out, (tuple, list)):
#                 out = out[0]

#             # Flip output back
#             if flip:
#                 out = torch.flip(out, dims=[-1])

#             # -------------------------------------------
#             # 4. Unpad & Resize back
#             # -------------------------------------------
#             # Crop padding
#             out = out[..., :new_h, :new_w]

#             # Resize back to original resolution (h, w)
#             if scale != 1.0:
#                 out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)

#             logits_sum += out
#             cnt += 1

#     return logits_sum / cnt

# # ============================================================
# # Main testing function
# # ============================================================
# def main(args):

#     # ---------------------------------------------------------
#     # Load YAML
#     # ---------------------------------------------------------
#     opt = parse(args.opt, is_train=False)
#     opt['dist'] = False

#     # ---------------------------------------------------------
#     # Create model
#     # ---------------------------------------------------------
#     model = create_model(opt).net_g

#     checkpoint = torch.load(args.weights)
#     try:
#         model.load_state_dict(checkpoint["params"])
#     except:
#         # fallback if weights have "module." prefix
#         model.load_state_dict({"module." + k: v for k, v in checkpoint["params"].items()})

#     model = model.cuda()
#     model.eval()
#     model = torch.nn.DataParallel(model)

#     print("✔ Loaded weights:", args.weights)

#     # ---------------------------------------------------------
#     # Dataset
#     # ---------------------------------------------------------
#     dataset_opt = opt["datasets"]["val"]
#     dataset_opt["phase"] = "test"

#     dataset = Dataset_PairedImage(dataset_opt)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

#     print("Dataset length:", len(dataset))

#     # ---------------------------------------------------------
#     # Prepare metrics
#     # ---------------------------------------------------------
#     num_classes = args.num_classes
#     factor = 32  # transformer size divisibility

#     os.makedirs(args.save_dir, exist_ok=True)

#     cm = np.zeros((num_classes, num_classes), dtype=np.int64)

#     # ---------------------------------------------------------
#     # Test Loop
#     # ---------------------------------------------------------
#     with torch.no_grad():

#         for batch in tqdm(dataloader):

#             torch.cuda.empty_cache()

#             x = batch["lq"].cuda()         # shape: (1,4,H,W)
#             gt = batch["gt"].squeeze(0).cpu().numpy().astype(np.int32)
#             name = os.path.splitext(os.path.basename(batch["lq_path"][0]))[0]
#             _, _, h, w = x.shape

#             # -------------------------------
#             # Forward pass (with TTA option)
#             # -------------------------------
#             if args.tta:
#                 # Pass RAW x. The function handles scaling/padding internally.
#                 logits = forward_tta(x, model, scales=[0.9, 1.0, 1.1])
#             else:
#                 # Standard single-scale path: Pad manually here
#                 H = (h + factor - 1) // factor * factor
#                 W = (w + factor - 1) // factor * factor
#                 pad_h = H - h
#                 pad_w = W - w
#                 x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
                
#                 logits = model(x_pad)
                
#                 # Handle tuple output
#                 if isinstance(logits, (tuple, list)):
#                     logits = logits[0]
                    
#                 # Remove padding
#                 logits = logits[:, :, :h, :w]

#             # -------------------------------
#             # Prediction
#             # -------------------------------
#             # logits is now [1, C, H, W]
#             print(logits.shape, gt.shape)
#             pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

#             # -------------------------------
#             # Update confusion matrix
#             # -------------------------------
#             cm = update_confusion_matrix(cm, pred, gt, num_classes)

#             # -------------------------------
#             # Save prediction color mask
#             # -------------------------------
#             color_mask = label2rgb(pred, bg_label=0)
#             color_mask = (color_mask * 255).astype(np.uint8)

#             cv2.imwrite(
#                 os.path.join(args.save_dir, f"{name}.png"),
#                 cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
#             )

#     # ---------------------------------------------------------
#     # Final global metrics
#     # ---------------------------------------------------------
#     miou, m_acc, pixacc, iou_per_class = compute_global_metrics(cm)

#     print("\n============== RESULTS ==============")
#     print("Mean IoU:", miou)
#     # print("Mean Acc:", m_acc)
#     print("Pixel Accuracy:", pixacc)
#     print("IoU per class:", iou_per_class)
#     print("=====================================")


# # ============================================================
# # Argument parser
# # ============================================================
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()

#     parser.add_argument("--opt", type=str, required=True, help="YAML config")
#     parser.add_argument("--weights", type=str, required=True, help="checkpoint .pth")
#     parser.add_argument("--save_dir", type=str, default="./results_seg/")
#     parser.add_argument("--num_classes", type=int, required=True)
#     parser.add_argument("--tta", action="store_true")

#     args = parser.parse_args()
#     main(args)

# PST900 and MFNet eval end

# # CART eval start

"""
Multi-Modal Segmentation Testing Script
- Inputs: RGB + Thermal
- Output: semantic logits
- Evaluation: mIoU (global), pixel accuracy
- Dataset: Paired RGB + Thermal + GT masks
- Logic: CART Dataset specific (Ignore 0,1; Remap 2-11 -> 0-9)
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
from basicsr.data.paired_image_dataset import Dataset_PairedImage


# ============================================================
# Utility: Update confusion matrix (Modified for CART)
# ============================================================
def update_confusion_matrix(cm, pred, gt, num_classes):
    """
    pred: H×W, int (Predicted classes, expected 0-9)
    gt:   H×W, int (Ground Truth IDs, expected 0-11)
    
    Logic applied:
    1. Ignore GT IDs 0 and 1.
    2. Remap GT IDs 2-11 to 0-9.
    3. Filter out invalid predictions if necessary.
    """
    pred = pred.flatten()
    gt = gt.flatten()

    # --- 1. Mask valid pixels (Raw IDs 2 through 11) ---
    valid_mask = (gt >= 2) & (gt <= 11)
    
    # Apply mask immediately to reduce computation on invalid pixels
    pred = pred[valid_mask]
    gt = gt[valid_mask]

    # --- 2. Remap target: 2->0, 3->1 ... 11->9 ---
    gt = gt - 2

    # --- 3. Safety Clean-up ---
    # Ensure both pred and gt are within the expected range [0, num_classes)
    # This handles cases where model might predict a class outside range 
    # or if remapping resulted in unexpected values.
    mask_safe = (pred >= 0) & (pred < num_classes) & (gt >= 0) & (gt < num_classes)
    
    pred = pred[mask_safe]
    gt = gt[mask_safe]

    # Update Global Confusion Matrix
    # Labels must be explicit to ensure shape is (num_classes, num_classes)
    cm += confusion_matrix(gt, pred, labels=list(range(num_classes)))
    return cm


# ============================================================
# Compute global metrics after processing entire dataset
# ============================================================
def compute_global_metrics(cm):
    """
    Calculates mIoU based on the accumulated Confusion Matrix.
    """
    intersection = np.diag(cm)
    union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - intersection
    
    # Calculate IoU per class
    # Use 1e-10 to avoid division by zero
    iou = intersection / np.maximum(union, 1e-10)
    
    # Standard mIoU
    miou = np.mean(iou)

    # Pixel Accuracy
    pixel_acc = intersection.sum() / (cm.sum() + 1e-10)
    m_acc = np.mean(pixel_acc) # This is usually just global pixel acc, but keeping variable name
    
    return miou, m_acc, pixel_acc, iou


# ============================================================
# Optional: Test-time augmentation (horizontal flip)
# ============================================================
def forward_tta(x, model, scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75]):
    b, c, h, w = x.shape
    logits_sum = 0
    cnt = 0

    for scale in scales:
        # Resize
        if scale != 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            x_s = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
        else:
            x_s = x
            new_h, new_w = h, w

        # Pad
        factor = 32
        H_pad = (new_h + factor - 1) // factor * factor
        W_pad = (new_w + factor - 1) // factor * factor
        pad_h = H_pad - new_h
        pad_w = W_pad - new_w
        
        x_s_pad = F.pad(x_s, (0, pad_w, 0, pad_h), mode='reflect')

        # Flip TTA
        for flip in [False, True]:
            input_tensor = torch.flip(x_s_pad, dims=[-1]) if flip else x_s_pad

            out = model(input_tensor)
            
            if isinstance(out, (tuple, list)):
                out = out[0]

            if flip:
                out = torch.flip(out, dims=[-1])

            # Unpad
            out = out[..., :new_h, :new_w]

            # Resize back
            if scale != 1.0:
                out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)

            logits_sum += out
            cnt += 1

    return logits_sum / cnt


# ============================================================
# Main testing function
# ============================================================
def main(args):
    # ---------------------------------------------------------
    # Load YAML and Model
    # ---------------------------------------------------------
    opt = parse(args.opt, is_train=False)
    opt['dist'] = False

    model = create_model(opt).net_g

    checkpoint = torch.load(args.weights)
    raw_state_dict = checkpoint["params"]
    
    # --- Fix: Strip "module." prefix if present ---
    # This aligns the checkpoint keys with your model's keys
    cleaned_state_dict = {}
    for k, v in raw_state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        cleaned_state_dict[name] = v

    # Load with strict=False to capture the lists
    load_status = model.load_state_dict(cleaned_state_dict, strict=False)

    model = model.cuda()
    model.eval()
    model = torch.nn.DataParallel(model)

    # ---------------------------------------------------------
    # Parameter Report (First 5 of each)
    # ---------------------------------------------------------
    print(f"\n✔ Loaded weights: {args.weights}")
    
    # 1. Found Keys (Intersection)
    model_keys = set(model.module.state_dict().keys())
    ckpt_keys = set(cleaned_state_dict.keys())
    found_keys = list(model_keys.intersection(ckpt_keys))
    
    print(f"\n✅ FOUND PARAMS (Total: {len(found_keys)}):")
    for k in found_keys[:5]:
        print(f"  - {k}")
    if len(found_keys) > 5: print("  ... (and more)")

    # 2. Missing Keys (In model but not in checkpoint)
    print(f"\n❌ MISSING PARAMS (Total: {len(load_status.missing_keys)}):")
    for k in load_status.missing_keys[:5]:
        print(f"  - {k}")
    if len(load_status.missing_keys) > 5: print("  ... (and more)")

    # 3. Supplementary/Unexpected Keys (In checkpoint but not in model)
    print(f"\n➕ SUPPLEMENTARY PARAMS (Total: {len(load_status.unexpected_keys)}):")
    for k in load_status.unexpected_keys[:5]:
        print(f"  - {k}")
    if len(load_status.unexpected_keys) > 5: print("  ... (and more)")
    
    print("-" * 50)

    # ... rest of your dataset and testing code ...

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------
    dataset_opt = opt["datasets"]["val"]
    dataset_opt["phase"] = "test"

    dataset = Dataset_PairedImage(dataset_opt)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    print("Dataset length:", len(dataset))

    # ---------------------------------------------------------
    # Prepare metrics
    # ---------------------------------------------------------
    # Ensure this is set to 10 for CART (Classes 0-9)
    num_classes = args.num_classes 
    factor = 32 

    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize Global Confusion Matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    # ---------------------------------------------------------
    # Test Loop
    # ---------------------------------------------------------
    with torch.no_grad():

        for batch in tqdm(dataloader):
            torch.cuda.empty_cache()

            x = batch["lq"].cuda()         # shape: (1,4,H,W)
            # GT loaded as int32
            gt = batch["gt"].squeeze(0).cpu().numpy().astype(np.int32)
            name = os.path.splitext(os.path.basename(batch["lq_path"][0]))[0]
            _, _, h, w = x.shape

            # -------------------------------
            # Forward pass
            # -------------------------------
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

            # -------------------------------
            # Prediction
            # -------------------------------
            # pred is likely 0-9 if model is trained on CART remapped classes
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

            # -------------------------------
            # Update confusion matrix (Apply Mask & Remap)
            # -------------------------------
            cm = update_confusion_matrix(cm, pred, gt, num_classes)

            # -------------------------------
            # Save prediction color mask
            # -------------------------------
            # Note: pred contains 0-9. Class 0 (Road) will be bg_label in label2rgb 
            # if not handled, but strictly for evaluation, this is fine.
            color_mask = label2rgb(pred, bg_label=0)
            color_mask = (color_mask * 255).astype(np.uint8)

            cv2.imwrite(
                os.path.join(args.save_dir, f"{name}.png"),
                cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
            )

    # ---------------------------------------------------------
    # Final global metrics
    # ---------------------------------------------------------
    miou, m_acc, pixacc, iou_per_class = compute_global_metrics(cm)

    print("\n============== RESULTS ==============")
    print(f"Evaluated on {num_classes} classes (Valid IDs 2-11 mapped to 0-9)")
    print("Mean IoU:      {:.4f}".format(miou))
    print("Pixel Accuracy: {:.4f}".format(pixacc))
    print("IoU per class:")
    for i, score in enumerate(iou_per_class):
        print(f"  Class {i}: {score:.4f}")
    print("=====================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, required=True, help="YAML config")
    parser.add_argument("--weights", type=str, required=True, help="checkpoint .pth")
    parser.add_argument("--save_dir", type=str, default="./results_seg/")
    
    # Ensure you pass 10 here for CART
    parser.add_argument("--num_classes", type=int, required=True) 
    parser.add_argument("--tta", action="store_true")

    args = parser.parse_args()
    main(args)


# CART eval end