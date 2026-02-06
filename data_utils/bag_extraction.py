import os
import cv2
import numpy as np
from tqdm import tqdm
import rosbag
from cv_bridge import CvBridge

def extract_vivid_bag_raw_stats(bag_path, output_dir):
    """
    Extracts RGB, Thermal, and Depth frames from a VIVID .bag file.
    Saves RGB/thermal as PNG, depth as PNG + NPY.
    Computes and prints raw min/max/mean/std per category BEFORE any normalization.
    """
    os.makedirs(output_dir, exist_ok=True)
    rgb_dir = os.path.join(output_dir, "rgb")
    thermal_dir = os.path.join(output_dir, "thermal")
    depth_png_dir = os.path.join(output_dir, "depth_png")
    depth_npy_dir = os.path.join(output_dir, "depth_npy")
    for d in [rgb_dir, thermal_dir, depth_png_dir, depth_npy_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"Reading bag file: {bag_path}")
    bag = rosbag.Bag(bag_path, "r")
    bridge = CvBridge()

    topic_map = {
        "/rgb/image": rgb_dir,
        "/thermal/image_raw": thermal_dir,
        "/depth/image_raw": (depth_png_dir, depth_npy_dir),
    }

    # Accumulators for statistics
    stats = {
        "rgb": [],
        "thermal": [],
        "depth": []
    }

    for topic, msg, t in tqdm(bag.read_messages(), desc="Extracting frames"):
        if topic not in topic_map:
            continue

        try:
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            timestamp = t.to_nsec()

            # Determine category and save
            if topic == "/rgb/image":
                cat = "rgb"
                filename = os.path.join(rgb_dir, f"rgb_{timestamp}.png")
                cv2.imwrite(filename, cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR))
            elif topic == "/thermal/image_raw":
                cat = "thermal"
                filename = os.path.join(thermal_dir, f"thermal_{timestamp}.png")
                cv2.imwrite(filename, cv_img)
            elif topic == "/depth/image_raw":
                cat = "depth"
                filename_png = os.path.join(depth_png_dir, f"depth_{timestamp}.png")
                filename_npy = os.path.join(depth_npy_dir, f"depth_{timestamp}.npy")
                # Save raw NPY
                np.save(filename_npy, cv_img)
                # Save PNG for visualization (normalized)
                cv_img_norm = cv_img
                if cv_img.dtype != np.uint8:
                    cv_img_norm = (cv_img / np.max(cv_img) * 255).astype(np.uint8)
                cv2.imwrite(filename_png, cv_img_norm)
            else:
                continue

            # Save raw values for statistics
            stats[cat].append(cv_img.astype(np.float32))

        except Exception as e:
            print(f"⚠️ Error decoding frame from {topic}: {e}")

    bag.close()

    # Compute raw statistics per category
    for cat, images in stats.items():
        if images:
            all_pixels = np.concatenate([img.flatten() for img in images])
            print(f"\n[{cat.upper()}] frames: {len(images)}")
            print(f"RAW min: {np.nanmin(all_pixels)}, max: {np.nanmax(all_pixels)}, "
                  f"mean: {np.nanmean(all_pixels):.3f}, std: {np.nanstd(all_pixels):.3f}")
        else:
            print(f"\n[{cat.upper()}] no frames found.")

    print(f"\n✅ Extraction completed: {output_dir}")


# Example usage
extract_vivid_bag_raw_stats(
    "/home/ancuti/Downloads/outdoor_robust_day1-002.bag",
    "/home/ancuti/expendables/Multeenhancer/data/VIVID_1/"
)
