#!/usr/bin/env python3
"""
generate_and_analyze.py

This script unifies the low-light synthetic image generation process
with a robust quality analysis. The '--output' argument is OPTIONAL.
If '--output' is not provided, images are NOT saved to disk, and analysis
is performed using in-memory data (dry run mode).

Parameters are tuned to align with average real-world dataset metrics 
(Exposure ~0.19, SNR ~8.9 db, NMI ~0.42) while maximizing light variability.
"""

import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from multiprocessing import Pool
import random 
import warnings
from statistics import stdev

# Suppress runtime warnings for cleaner output
warnings.filterwarnings("ignore")

# =========================================================
#  I. CORE LOGIC & METRICS (Generation & Analysis Tools)
# =========================================================

# --- Generation Core ---

def decompose_image(Y_channel):
    Y = Y_channel.astype(np.float32)
    base = cv2.GaussianBlur(Y, (0, 0), sigmaX=15, sigmaY=15)
    base = np.clip(base, 1.0, 255.0)
    detail = Y / base
    return base, detail

def histogram_specification(source_channel, target_cdf, target_bins):
    src_flat = source_channel.ravel()
    src_hist, _ = np.histogram(src_flat, bins=256, range=[0, 256], density=True)
    src_cdf = src_hist.cumsum()
    src_cdf /= src_cdf.max()
    lut = np.interp(src_cdf, target_cdf, target_bins)
    lut = np.clip(lut, 0, 240)
    src_indices = np.clip(src_flat, 0, 255).astype(np.uint8)
    return lut[src_indices].reshape(source_channel.shape).astype(np.float32)

def load_distribution(path):
    try:
        data = np.load(path, allow_pickle=True).item()
    except:
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.ndarray): data = data.item()
    if 'cdf' in data:
        cdf = data['cdf']
    elif 'pdf' in data:
        cdf = np.cumsum(data['pdf'])
        cdf /= cdf.max()
    else:
        raise ValueError("File must contain 'cdf' or 'pdf'.")
    bins = data.get('bins', np.arange(256))
    return cdf, bins

def add_tunable_noise(image, gain=1.0, shot_noise_scale=300.0, read_noise_sigma=0.01):
    if gain <= 0.0: return image
    
    img_linear = np.power(image.astype(np.float32) / 255.0, 2.2)
    img_linear = np.clip(img_linear, 1e-6, 1.0)
    
    poisson_noisy = np.random.poisson(img_linear * shot_noise_scale) / shot_noise_scale
    poisson_delta = poisson_noisy - img_linear
    gaussian_delta = np.random.normal(loc=0, scale=read_noise_sigma, size=img_linear.shape)
    
    total_noise = (poisson_delta + gaussian_delta) * gain
    noisy_linear = img_linear + total_noise
    
    noisy_linear = np.clip(noisy_linear, 0.0, 1.0)
    img_srgb = np.power(noisy_linear, 1.0/2.2)
    
    return (img_srgb * 255).astype(np.uint8)

def remove_hot_pixels(img, threshold=250):
    mask = img > threshold
    if not np.any(mask): return img
    median = cv2.medianBlur(img, 3)
    return np.where(mask, median, img).astype(np.uint8)

# --- Analysis Metrics (Moved from separate script) ---

def calculate_snr(signal_img, noisy_img):
    s = signal_img.astype(np.float64)
    n = noisy_img.astype(np.float64)
    mean_s = np.mean(s)
    if mean_s < 1e-6: return None
    mu_signal = np.mean(n)
    scale = mu_signal / mean_s
    s_scaled = s * scale
    noise_map = n - s_scaled
    sigma_noise = np.std(noise_map)
    if sigma_noise < 1e-9: return 100.0
    snr_db = 20 * np.log10(mu_signal / sigma_noise)
    return snr_db

def calculate_nmi(img1, img2):
    bins = 64
    hist_2d, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=bins)
    pxy = hist_2d / float(np.sum(hist_2d) + 1e-9)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    h_x = -np.sum(px[px > 0] * np.log(px[px > 0]))
    h_y = -np.sum(py[py > 0] * np.log(py[py > 0]))
    if (h_x + h_y) == 0: return 0.0
    return 2 * mi / (h_x + h_y)

def estimate_physics_params(normal_img, low_img):
    N = normal_img.astype(np.float32) / 255.0
    L = low_img.astype(np.float32) / 255.0
    epsilon = 1e-6
    mean_n = np.mean(N)
    mean_l = np.mean(L)
    if mean_n < epsilon: return None, None
    E_est = mean_l / mean_n
    mask = (N > 0.05) & (N < 0.95) & (L > epsilon)
    if np.sum(mask) < 50: return E_est, 1.0 
    val_N = N[mask]
    val_L = L[mask]
    numerator = np.log(val_L)
    denominator = np.log(val_N * E_est + epsilon)
    valid = np.abs(denominator) > 1e-4
    if np.sum(valid) == 0: return E_est, 1.0
    gammas = numerator[valid] / denominator[valid]
    gamma_est = np.median(gammas)
    return E_est, np.clip(gamma_est, 0.1, 5.0)

# =========================================================
#  II. WORKER FUNCTION (Generates and Calculates Metrics)
# =========================================================

def process_single_image(args):
    img_path, dist_cdf, dist_bins, structure_bias, noise_gain, exposure_scale, output_dir = args
    filename = os.path.basename(img_path)

    try:
        # 1. Load Original Image (N)
        img_orig = cv2.imread(img_path)
        if img_orig is None: raise ValueError("Could not read image.")
        Y_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2YCrCb)[:,:,0]

        # 2. GENERATION Pipeline
        ycrcb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2YCrCb)
        Y_local = ycrcb[:, :, 0].astype(np.float32)
        Cr = ycrcb[:, :, 1]
        Cb = ycrcb[:, :, 2]
        base, detail = decompose_image(Y_local)
        base_matched = histogram_specification(base, dist_cdf, dist_bins)
        Y_retinex = base_matched * detail
        avg_scale = np.mean(base_matched) / (np.mean(base) + 1e-6)
        Y_linear = Y_local * avg_scale
        highlight_mask = np.clip((Y_local - 150) / 100.0, 0.0, 1.0)
        final_mask = np.maximum(highlight_mask, structure_bias)
        Y_final = (final_mask * Y_linear) + ((1.0 - final_mask) * Y_retinex)
        Y_final = np.clip(Y_final, 0, 245).astype(np.uint8)
        merged = cv2.merge([Y_final, Cr, Cb])
        img_low = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
        if exposure_scale != 1.0:
            img_low = (img_low.astype(np.float32) * exposure_scale)
            img_low = np.clip(img_low, 0, 255).astype(np.uint8)
        img_noisy = add_tunable_noise(img_low, gain=noise_gain, shot_noise_scale=300.0, read_noise_sigma=0.01)
        img_final = remove_hot_pixels(img_noisy, threshold=250)
        
        # 3. Save to Disk (CONDITIONAL)
        if output_dir:
            save_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".png")
            cv2.imwrite(save_path, img_final)

        # 4. Calculate Metrics (L)
        Y_syn = cv2.cvtColor(img_final, cv2.COLOR_BGR2YCrCb)[:,:,0]
        
        snr = calculate_snr(Y_orig, Y_syn)
        if snr is None: return None # Skip if SNR calculation failed (e.g., zero signal)
        
        nmi = calculate_nmi(Y_orig, Y_syn)
        E, G = estimate_physics_params(Y_orig, Y_syn)
        if E is None: E, G = 0.0, 1.0

        # 5. Return Results to Main Thread
        return {
            'snr': snr,
            'nmi': nmi,
            'exp': E,
            'gamma': G,
            'lum_y': np.mean(Y_syn), # Mean Luminance of this image
        }

    except Exception as e:
        return None

# =========================================================
#  III. UNIFIED MAIN FUNCTION
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic low-light images and analyze quality metrics. Output is optional.")
    parser.add_argument('--input', type=str, required=True, help='Folder with Normal Light images (Original Dir)')
    parser.add_argument('--dist_file', type=str, required=True, help='Path to the Real Low-Light Distribution file (.npy)')
    
    # 🌟 MODIFIED: OUTPUT IS NOW OPTIONAL (required=False) 🌟
    parser.add_argument('--output', type=str, required=False, help='Output Folder for Synthetic Low Light images (Synthetic Dir). Omit this argument for a dry run/in-memory analysis.')
    
    # 🌟 Tuned for Alignment (Avg Real World: NMI 0.42) 🌟
    parser.add_argument('--structure_bias', type=float, default=0.8, 
                        help='Structure Weight. Default 0.8 to heavily preserve structure and increase NMI towards 0.42.')
    
    # 🌟 Tuned for Alignment (Avg Real World: SNR 8.93 db) 🌟
    parser.add_argument('--noise_gain', type=float, default=0.3, 
                        help='Noise Multiplier. Default 0.3 to reduce noise and increase SNR towards 8.93 db.')
    
    # 🌟 Tuned for Alignment (Avg Real World: Exposure 0.19) and High CV 🌟
    parser.add_argument('--exposure_range', type=float, nargs=2, default=[0.05, 0.35], 
                        help='Min and Max for random global dimming factor. Default 0.05 0.35 to align mean exposure with 0.19 and maintain high variability.')
    
    parser.add_argument('--workers', type=int, default=4, help='Number of multiprocessing workers.')
    args = parser.parse_args()

    # --- Setup ---
    is_saving = bool(args.output)
    if is_saving:
        os.makedirs(args.output, exist_ok=True)
    
    print("\n" + "="*50)
    print("PHASE 1: SYNTHETIC DATA GENERATION & METRICS CALCULATION")
    print("="*50)
    
    print(f"Loading distribution from {args.dist_file}...")
    dist_cdf, dist_bins = load_distribution(args.dist_file)

    exts = ('.jpg', '.png', '.bmp', '.tif')
    files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith(exts)]
    
    print(f"Processing {len(files)} images. Output to disk: {is_saving}")
    print(f"Settings: Bias={args.structure_bias}, Noise={args.noise_gain}, Exposure Range={args.exposure_range}")
    
    # --- Prepare Worker Arguments ---
    min_exp, max_exp = args.exposure_range
    worker_args = []
    
    for f in files:
        random_exposure = random.uniform(min_exp, max_exp)
        output_arg = args.output if is_saving else None 
        worker_args.append((f, dist_cdf, dist_bins, args.structure_bias, args.noise_gain, random_exposure, output_arg))
    
    # --- Run Generation and Collect Metrics ---
    
    all_metrics = []
    
    with Pool(args.workers) as pool:
        results = list(tqdm(pool.imap(process_single_image, worker_args), total=len(files), desc="Generating & Analyzing"))
        all_metrics = [r for r in results if r is not None]

    # --- PHASE 2: REPORTING (CLEANED) ---
    print("\n" + "="*50)
    print("SYNTHETIC DATA ANALYSIS REPORT")
    print("="*50)
    
    num_valid = len(all_metrics)
    if num_valid > 0:
        
        # Aggregate metrics
        snr_list = [m['snr'] for m in all_metrics]
        nmi_list = [m['nmi'] for m in all_metrics]
        exp_list = [m['exp'] for m in all_metrics]
        gamma_list = [m['gamma'] for m in all_metrics]
        lum_y_list = [m['lum_y'] for m in all_metrics] # List of Mean Luminance for every image
        
        # Luminance Statistics
        mean_lum_y = np.mean(lum_y_list)
        std_lum_y = stdev(lum_y_list) if num_valid > 1 else 0
        cv_lum_y = (std_lum_y / mean_lum_y) * 100 if mean_lum_y > 1e-6 else 0
        
        # 🌟 CORRECTED: Min/Max of the Mean Luminance List (lum_y_list) 🌟
        min_avg_lum_y = np.min(lum_y_list)
        max_avg_lum_y = np.max(lum_y_list)
        
        print(f"Valid Pairs Processed: {num_valid} / {len(files)}")
        print("-" * 50)
        print(f"Mean Exposure (E):           {np.mean(exp_list):.4f}")
        print(f"Mean SNR (dB):               {np.mean(snr_list):.2f}")
        print(f"Mean NMI:                    {np.mean(nmi_list):.4f}")
        print(f"Mean Gamma:                  {np.mean(gamma_list):.4f}")
        print("-" * 50)
        print(f"Luminance Variability (CV):  {cv_lum_y:.2f}%")
        print(f"Avg Luminance (Y 0-255):     {mean_lum_y:.2f}")
        print(f"Std Dev Luminance (Y):       {std_lum_y:.2f}")
        print(f"Min Avg Luminance (Y 0-255): {min_avg_lum_y:.7f}") 
        print(f"Max Avg Luminance (Y 0-255): {max_avg_lum_y:.7f}")
    else:
        print("NO VALID DATA COULD BE ANALYZED.")

    print("="*50)

if __name__ == "__main__":
    main()