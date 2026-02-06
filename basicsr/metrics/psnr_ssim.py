import cv2
import numpy as np

from basicsr.metrics.metric_util import reorder_image, to_y_channel
import skimage.metrics
import torch


def calculate_psnr(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray/tensor): Images with range [0, 255]/[0, 1].
        img2 (ndarray/tensor): Images with range [0, 255]/[0, 1].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    if type(img1) == torch.Tensor:
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1,2,0)
    if type(img2) == torch.Tensor:
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1,2,0)
        
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)
    
    # mean_restored = cv2.cvtColor(img2.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
    # mean_target = cv2.cvtColor(img1.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
    # img1 = np.clip(img1 * (mean_target / mean_restored), 0, 1)

    # mse = np.mean((img1 - img2)**2)
    # if mse == 0:
    #     return float('inf')
    # max_value = 1. if img1.max() <= 1 else 255.
    # return 20. * np.log10(max_value / np.sqrt(mse))
    max_value = 1. if img1.max() <= 1 else 255.
    img1_gray = img1.mean(axis=0)
    img2_gray = img2.mean(axis=0)
    mean_pred = img1_gray.mean()
    mean_target = img2_gray.mean()
    # img1 = np.clip(img1 * (mean_target / mean_pred), 0, max_value)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(max_value / np.sqrt(mse))

# MFNet calculation
def calculate_miou(pred, target, num_classes=5):
    import torch
    import numpy as np
    
    if isinstance(pred, torch.Tensor):
        # [B, C, H, W] → [C, H, W]
        if pred.dim() == 4:
            pred = pred.squeeze(0)
        # [C, H, W] → [H, W]
        if pred.dim() == 3:
            pred = torch.argmax(pred, dim=0)
        pred = pred.detach().cpu().numpy()
    
    if isinstance(target, torch.Tensor):
        if target.dim() == 4:
            target = target.squeeze(0)
        target = target.squeeze(0)  # handles [1,H,W] or [H,W]
        target = target.detach().cpu().numpy()
    
    pred = pred.astype(np.int64)
    target = target.astype(np.int64)
    
    ious = []
    for classy in range(num_classes):
        pred_mask = pred == classy
        target_mask = target == classy
        
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        
        if union == 0:
            # Class not present in ground truth or prediction
            # Only count as perfect if both pred and target don't have this class
            if intersection == 0:
                ious.append(1.0)  # True negative - correctly predicted absence
            else:
                ious.append(0.0)  # Should not happen if union==0 and intersection==0
        else:
            iou = intersection / union
            ious.append(iou)
    
    return np.mean(ious)

# MFNet calcualtion

# CART calculation

# import torch
# import numpy as np

# def calculate_miou(pred, target, num_classes=10):
#     """
#     Calculates mIoU for CART dataset.
#     - Ignores IDs 0 (Unknown) and 1 (Background) in the target.
#     - Remaps target IDs 2-11 to 0-9.
#     - Excludes ignored pixels from False Positive calculation.
#     """
    
#     # --- 1. Tensor to Numpy & Shape Standardization ---
#     if isinstance(pred, torch.Tensor):
#         if pred.dim() == 4: # [B, C, H, W] -> [B, H, W]
#             pred = torch.argmax(pred, dim=1)
#         # Flatten batch dimension for simple global calculation over the batch
#         pred = pred.detach().cpu().numpy()

#     if isinstance(target, torch.Tensor):
#         # Handle cases where target is [B, 1, H, W] or [B, H, W]
#         if target.dim() == 4:
#             target = target.squeeze(1)
#         target = target.detach().cpu().numpy()
    
#     # Ensure they are integers
#     pred = pred.astype(np.int64)
#     target = target.astype(np.int64)

#     # --- 2. CART Specific Logic: Remap & Mask ---
#     # Create a mask of valid pixels (Raw IDs 2 through 11)
#     valid_mask = (target >= 2) & (target <= 11)
    
#     # Remap target: 2->0, 3->1, ... 11->9
#     # We copy target to avoid modifying the original array outside function
#     target_remapped = target.copy()
#     target_remapped[valid_mask] = target_remapped[valid_mask] - 2
    
#     # Set invalid pixels in target to a safe ignore value (e.g., -1)
#     target_remapped[~valid_mask] = -1

#     # --- 3. Calculate IoU ---
#     ious = []
#     for classy in range(num_classes):
#         # We only consider pixels that are VALID in the ground truth.
#         # This prevents "Background" pixels from counting as False Positives.
        
#         # Pred mask: Model predicts Class X AND pixel is valid
#         pred_mask = (pred == classy) & valid_mask
        
#         # Target mask: Target is Class X AND pixel is valid
#         target_mask = (target_remapped == classy) & valid_mask
        
#         intersection = np.logical_and(pred_mask, target_mask).sum()
#         union = np.logical_or(pred_mask, target_mask).sum()
        
#         if union == 0:
#             # Class not present in ground truth OR prediction (within valid regions)
#             # If intersection is also 0, it means we correctly predicted its absence.
#             if intersection == 0:
#                 ious.append(1.0) 
#             else:
#                 # This case is mathematically impossible if intersection is subset of union
#                 ious.append(0.0) 
#         else:
#             iou = intersection / union
#             ious.append(iou)
    
#     return np.mean(ious)

# CART calculation


def calculate_mae(img1,
                  img2,
                  crop_border,
                  input_order='HWC',
                  test_y_channel=False):
    """Calculate Mean Absolute Error (MAE) between two images.

    Args:
        img1 (ndarray/tensor): Image with range [0, 255]/[0, 1].
        img2 (ndarray/tensor): Image with range [0, 255]/[0, 1].
        crop_border (int): Cropped pixels in each edge of the image.
            These pixels are not involved in the MAE calculation.
        input_order (str): 'HWC' or 'CHW'. Default: 'HWC'.
        test_y_channel (bool): Whether to compute MAE on Y channel of YCbCr.
            Default: False.

    Returns:
        float: MAE result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are different: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW".')

    # --- Convert to numpy if torch.Tensor ---
    if isinstance(img1, torch.Tensor):
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    if isinstance(img2, torch.Tensor):
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)

    # --- Reorder and crop ---
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    # --- Optional Y channel ---
    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    # --- Compute MAE ---
    mae = np.mean(np.abs(img1 - img2))

    return float(mae)






import torch
import numpy as np

def calculate_absrel(img1,
                     img2,
                     crop_border,
                     input_order='HWC',
                     test_y_channel=False,
                     eps=1e-6):
    """Calculate Absolute Relative Error (AbsRel) between two images (e.g. depth maps).

    AbsRel = mean( |pred - gt| / (gt + eps) )

    Args:
        img1 (ndarray/tensor): Predicted image with range [0, 255]/[0, 1].
        img2 (ndarray/tensor): Ground-truth image with range [0, 255]/[0, 1].
        crop_border (int): Cropped pixels in each edge of the image.
        input_order (str): 'HWC' or 'CHW'. Default: 'HWC'.
        test_y_channel (bool): Whether to compute on Y channel of YCbCr.
        eps (float): Small constant to avoid division by zero.

    Returns:
        float: Absolute Relative Error.
    """

    print(
            f"METRIC: pred: [{img1.min().item():.4f}, {img1.max().item():.4f}], "
            f"METRIC: target: [{img2.min().item():.4f}, {img2.max().item():.4f}]"
        )

    assert img1.shape == img2.shape, (
        f'Image shapes are different: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW".')

    # --- Convert to numpy if torch.Tensor ---
    if isinstance(img1, torch.Tensor):
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    if isinstance(img2, torch.Tensor):
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)

    # --- Reorder and crop ---
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    # --- Optional Y channel ---
    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    # --- Mask out invalid gt (depth = 0 or NaN) ---
    # --- Detect invalid ground-truth maps ---
    if np.all((img2 <= eps) | ~np.isfinite(img2)):
        print(f"⚠️ Skipping sample — invalid depth! Range: [{img2.min():.4f}, {img2.max():.4f}]")
        return 0.0   # <--- return neutral value instead of NaN

    # --- Mask out invalid pixels ---
    valid_mask = (img2 > eps) & np.isfinite(img2)
    if np.sum(valid_mask) == 0:
        print("⚠️ No valid depth pixels! img2 range:", img2.min(), img2.max())
        return 0.0   # <--- same neutral return

    # --- Compute AbsRel ---
    absrel = np.mean(np.abs(img1[valid_mask] - img2[valid_mask]) / (img2[valid_mask] + eps))
    if np.isnan(absrel) or np.isinf(absrel):
        print("⚠️ Invalid AbsRel value encountered.")
        return 0.0

    return float(absrel)


def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def prepare_for_ssim(img, k):
    import torch
    with torch.no_grad():
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
        conv = torch.nn.Conv2d(1, 1, k, stride=1, padding=k//2, padding_mode='reflect')
        conv.weight.requires_grad = False
        conv.weight[:, :, :, :] = 1. / (k * k)

        img = conv(img)

        img = img.squeeze(0).squeeze(0)
        img = img[0::k, 0::k]
    return img.detach().cpu().numpy()

def prepare_for_ssim_rgb(img, k):
    import torch
    with torch.no_grad():
        img = torch.from_numpy(img).float() #HxWx3

        conv = torch.nn.Conv2d(1, 1, k, stride=1, padding=k // 2, padding_mode='reflect')
        conv.weight.requires_grad = False
        conv.weight[:, :, :, :] = 1. / (k * k)

        new_img = []

        for i in range(3):
            new_img.append(conv(img[:, :, i].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)[0::k, 0::k])

    return torch.stack(new_img, dim=2).detach().cpu().numpy()

def _3d_gaussian_calculator(img, conv3d):
    out = conv3d(img.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    return out

def _generate_3d_gaussian_kernel():
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    kernel_3 = cv2.getGaussianKernel(11, 1.5)
    kernel = torch.tensor(np.stack([window * k for k in kernel_3], axis=0))
    conv3d = torch.nn.Conv3d(1, 1, (11, 11, 11), stride=1, padding=(5, 5, 5), bias=False, padding_mode='replicate')
    conv3d.weight.requires_grad = False
    conv3d.weight[0, 0, :, :, :] = kernel
    return conv3d

def _ssim_3d(img1, img2, max_value):
    assert len(img1.shape) == 3 and len(img2.shape) == 3
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255]/[0, 1] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255]/[0, 1] with order 'HWC'.

    Returns:
        float: ssim result.
    """
    C1 = (0.01 * max_value) ** 2
    C2 = (0.03 * max_value) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = _generate_3d_gaussian_kernel().cuda()

    img1 = torch.tensor(img1).float().cuda()
    img2 = torch.tensor(img2).float().cuda()


    mu1 = _3d_gaussian_calculator(img1, kernel)
    mu2 = _3d_gaussian_calculator(img2, kernel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = _3d_gaussian_calculator(img1 ** 2, kernel) - mu1_sq
    sigma2_sq = _3d_gaussian_calculator(img2 ** 2, kernel) - mu2_sq
    sigma12 = _3d_gaussian_calculator(img1*img2, kernel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())

def _ssim_cly(img1, img2):
    assert len(img1.shape) == 2 and len(img2.shape) == 2
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    # print(kernel)
    window = np.outer(kernel, kernel.transpose())

    bt = cv2.BORDER_REPLICATE

    mu1 = cv2.filter2D(img1, -1, window, borderType=bt)
    mu2 = cv2.filter2D(img2, -1, window,borderType=bt)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window, borderType=bt) - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window, borderType=bt) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window, borderType=bt) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')

    if type(img1) == torch.Tensor:
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1,2,0)
    if type(img2) == torch.Tensor:
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1,2,0)

    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)
        return _ssim_cly(img1[..., 0], img2[..., 0])


    ssims = []
    # ssims_before = []

    # skimage_before = skimage.metrics.structural_similarity(img1, img2, data_range=255., multichannel=True)
    # print('.._skimage',
    #       skimage.metrics.structural_similarity(img1, img2, data_range=255., multichannel=True))
    max_value = 1 if img1.max() <= 1 else 255
    with torch.no_grad():
        final_ssim = _ssim_3d(img1, img2, max_value)
        ssims.append(final_ssim)

    # for i in range(img1.shape[2]):
    #     ssims_before.append(_ssim(img1, img2))

    # print('..ssim mean , new {:.4f}  and before {:.4f} .... skimage before {:.4f}'.format(np.array(ssims).mean(), np.array(ssims_before).mean(), skimage_before))
        # ssims.append(skimage.metrics.structural_similarity(img1[..., i], img2[..., i], multichannel=False))

    return np.array(ssims).mean()
