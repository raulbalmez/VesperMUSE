import os
import random
import shutil

# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def copy_and_rename_consistent(rgb_files, thermal_files, seg_files, src_dirs, dst_dirs, start_idx=1):
    """
    Copy RGB, thermal, and segmentation files together and rename using the same counter.
    Ensures all intermediate directories exist.
    """
    assert len(rgb_files) == len(thermal_files) == len(seg_files), "Files must match in length"
    counter = start_idx
    for rgb, thermal, seg in zip(rgb_files, thermal_files, seg_files):
        # Get extensions
        ext_rgb = os.path.splitext(rgb)[1]
        ext_thermal = os.path.splitext(thermal)[1]
        ext_seg = os.path.splitext(seg)[1]

        new_name_rgb = f"{counter:05d}{ext_rgb}"
        new_name_thermal = f"{counter:05d}{ext_thermal}"
        new_name_seg = f"{counter:05d}{ext_seg}"

        # Ensure destination directories exist
        for dst in dst_dirs:
            ensure_dir(dst)

        # Copy files
        shutil.copy2(os.path.join(src_dirs[0], rgb), os.path.join(dst_dirs[0], new_name_rgb))
        shutil.copy2(os.path.join(src_dirs[1], thermal), os.path.join(dst_dirs[1], new_name_thermal))
        shutil.copy2(os.path.join(src_dirs[2], seg), os.path.join(dst_dirs[2], new_name_seg))

        counter += 1
    return counter

# -----------------------------
# CONFIG
# -----------------------------
DATASET_ROOT = "/home/ancuti/expendables/VesperMUSE/data/CART"
COLOR_DIR = os.path.join(DATASET_ROOT, "color")
THERMAL_DIR = os.path.join(DATASET_ROOT, "thermal8")
SEG_DIR = os.path.join(DATASET_ROOT, "annotations")

# -----------------------------
# READ FILES (sorted order)
# -----------------------------
color_files = sorted(os.listdir(COLOR_DIR))
thermal_files = sorted(os.listdir(THERMAL_DIR))
seg_files = sorted(os.listdir(SEG_DIR))

assert len(color_files) == len(thermal_files) == len(seg_files), \
    "Color, thermal, and segmentation folders must have the same number of files."

total = len(color_files)
print(f"[DEBUG] Total files: {total}")

# -----------------------------
# SPLIT INDICES
# -----------------------------
indices = list(range(total))
random.shuffle(indices)

train_n = int(total * 0.75)
val_n = int(total * 0.125)
test_n = total - train_n - val_n

train_idx = indices[:train_n]
val_idx = indices[train_n:train_n+val_n]
test_idx = indices[train_n+val_n:]

def select_files(idx_list, file_list):
    return [file_list[i] for i in idx_list]

# -----------------------------
# SPLIT FILES
# -----------------------------
train_color = select_files(train_idx, color_files)
train_thermal = select_files(train_idx, thermal_files)
train_seg = select_files(train_idx, seg_files)

val_color = select_files(val_idx, color_files)
val_thermal = select_files(val_idx, thermal_files)
val_seg = select_files(val_idx, seg_files)

test_color = select_files(test_idx, color_files)
test_thermal = select_files(test_idx, thermal_files)
test_seg = select_files(test_idx, seg_files)

# -----------------------------
# OUTPUT DIRECTORIES
# -----------------------------
TRAIN_OUT_COLOR = os.path.join(DATASET_ROOT, "train/color")
TRAIN_OUT_THERMAL = os.path.join(DATASET_ROOT, "train/thermal")
TRAIN_OUT_SEG = os.path.join(DATASET_ROOT, "train/segmentation")

TEST1_VAL_COLOR = os.path.join(DATASET_ROOT, "test/split1/val/color")
TEST1_VAL_THERMAL = os.path.join(DATASET_ROOT, "test/split1/val/thermal")
TEST1_VAL_SEG = os.path.join(DATASET_ROOT, "test/split1/val/segmentation")

TEST1_TEST_COLOR = os.path.join(DATASET_ROOT, "test/split1/test/color")
TEST1_TEST_THERMAL = os.path.join(DATASET_ROOT, "test/split1/test/thermal")
TEST1_TEST_SEG = os.path.join(DATASET_ROOT, "test/split1/test/segmentation")

TEST2_COMBINED_COLOR = os.path.join(DATASET_ROOT, "test/split2/combined/color")
TEST2_COMBINED_THERMAL = os.path.join(DATASET_ROOT, "test/split2/combined/thermal")
TEST2_COMBINED_SEG = os.path.join(DATASET_ROOT, "test/split2/combined/segmentation")

# -----------------------------
# COPY & RENAME
# -----------------------------
counter = 1
print("[DEBUG] Copying and renaming split1 train files...")
counter = copy_and_rename_consistent(
    train_color, train_thermal, train_seg,
    [COLOR_DIR, THERMAL_DIR, SEG_DIR],
    [TRAIN_OUT_COLOR, TRAIN_OUT_THERMAL, TRAIN_OUT_SEG],
    counter
)

print("[DEBUG] Copying and renaming split1 val files...")
counter = copy_and_rename_consistent(
    val_color, val_thermal, val_seg,
    [COLOR_DIR, THERMAL_DIR, SEG_DIR],
    [TEST1_VAL_COLOR, TEST1_VAL_THERMAL, TEST1_VAL_SEG],
    counter
)

print("[DEBUG] Copying and renaming split1 test files...")
counter = copy_and_rename_consistent(
    test_color, test_thermal, test_seg,
    [COLOR_DIR, THERMAL_DIR, SEG_DIR],
    [TEST1_TEST_COLOR, TEST1_TEST_THERMAL, TEST1_TEST_SEG],
    counter
)

# Split2: val + test combined (restart counter at 1)
print("[DEBUG] Copying and renaming split2 combined files...")
copy_and_rename_consistent(
    val_color + test_color,
    val_thermal + test_thermal,
    val_seg + test_seg,
    [COLOR_DIR, THERMAL_DIR, SEG_DIR],
    [TEST2_COMBINED_COLOR, TEST2_COMBINED_THERMAL, TEST2_COMBINED_SEG],
    start_idx=1
)

print("✔ Done. All splits created successfully with consistent numbering across modalities.")
