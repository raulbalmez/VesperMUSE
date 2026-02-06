import os
import glob
import cv2
import numpy as np
import configparser

class CartClassWeights:
    """
    Compute class weights for MFNet from TRAIN labels only.
    Ignores IDs 0 (Unknown) and 1 (Background), mapping IDs 2-11 to indices 0-9.
    """

    def __init__(self, datapath, num_classes=10):
        self.data_path = datapath
        self.label_path_train = os.path.join(datapath, "train", "segmentation")
        self.num_classes = num_classes
        self.weights = None
        self.label_paths = []

    def process_labels(self):
        """
        Loads all training label images and computes class weights.
        """
        self.label_paths = sorted(glob.glob(os.path.join(self.label_path_train, "*.png")))
        if len(self.label_paths) == 0:
            raise RuntimeError(f"No .png labels found at: {self.label_path_train}")

        print(f"Found {len(self.label_paths)} training labels. Loading data...")
        
        # Optimization: Use a list to avoid allocating massive memory at once if possible,
        # though np.stack will eventually consume it. 
        label_stack = []
        for p in self.label_paths:
            lbl = cv2.imread(p, -1)
            if lbl is None:
                raise RuntimeError(f"Cannot read label: {p}")
            label_stack.append(lbl)

        labels_np = np.stack(label_stack, axis=0)
        self.weights = self.calculate_class_weights(labels_np, self.num_classes)

        print("Class weights:", self.weights)
        return self.weights

    def calculate_class_weights(self, Y, n_classes, method="paszke", c=1.02):
        """
        Computes Paszke weights given integer label map.
        Specific logic for CART: Ignores 0, 1 and shifts 2..11 -> 0..9
        """
        print("Computing unique IDs and counts (this may take a moment)...")
        ids, counts = np.unique(Y, return_counts=True)
        print(f"Raw IDs found in dataset: {ids}")

        # Initialize counts for the 10 target classes (indices 0-9)
        mapped_counts = np.zeros(n_classes, dtype=np.float64)
        
        valid_pixel_total = 0

        for i, original_id in enumerate(ids):
            # LOGIC CHANGE: Ignore IDs 0 and 1
            if original_id < 2:
                continue
            
            # Map ID 2 -> Index 0, ID 11 -> Index 9
            mapped_index = original_id - 2
            
            if 0 <= mapped_index < n_classes:
                count = counts[i]
                mapped_counts[mapped_index] = count
                valid_pixel_total += count
            else:
                print(f"Warning: Found ID {original_id} which is out of expected range (2-11).")

        if valid_pixel_total == 0:
            raise RuntimeError("No valid pixels (IDs 2-11) found in the dataset!")

        # Calculate frequency based ONLY on valid pixels
        p_class = mapped_counts / valid_pixel_total

        # Paszke: 1 / log(c + p)
        weights = 1.0 / np.log(c + p_class)
        return weights

    def save_class_weights(self, weight_file):
        """
        Save weights to an .ini file.
        """
        if self.weights is None:
            raise RuntimeError("Compute weights first!")

        config = configparser.ConfigParser()
        config["ClassWeights"] = {}

        # IDs 2 through 11 map to these names in order
        class_names = [
            "bare_ground",    # ID 2
            "rocky_terrain",  # ID 3
            "devel_struct",   # ID 4
            "road",           # ID 5
            "shrubs",         # ID 6
            "trees",          # ID 7
            "sky",            # ID 8
            "water",          # ID 9
            "vehicles",       # ID 10
            "person"          # ID 11
        ]

        for i, name in enumerate(class_names):
            if i < len(self.weights):
                config["ClassWeights"][name] = str(self.weights[i])

        config["ClassWeights"]["num_images"] = str(len(self.label_paths))

        with open(weight_file, "w") as f:
            config.write(f)

        print(f"Saved MFNet weights to: {weight_file}")

    def load_class_weights(self, weight_file):
        config = configparser.ConfigParser()
        config.read(weight_file)

        class_names = ["bare_ground", "rocky_terrain", "devel_struct", "road", "shrubs", "trees", "sky", "water", "vehicles", "person"]

        weights = np.zeros(self.num_classes)
        for i, name in enumerate(class_names):
            weights[i] = float(config["ClassWeights"][name])

        num_images = int(config["ClassWeights"]["num_images"])
        return weights, num_images


def compute_mfnet_weights(path, save=True):
    cw = CartClassWeights(path, num_classes=10)
    weights = cw.process_labels()

    if save:
        out_file = os.path.join(path, "cart_weights.ini")
        cw.save_class_weights(out_file)

    return weights


if __name__ == "__main__":
    # Example:
    mfnet_path = "/home/ancuti/expendables/Multeenhancer/data/CART/"
    compute_mfnet_weights(mfnet_path, save=True)