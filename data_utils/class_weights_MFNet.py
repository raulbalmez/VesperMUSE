import os
import glob
import cv2
import numpy as np
import configparser


class MFNetClassWeights:
    """
    Compute class weights for MFNet from TRAIN labels only.
    """

    def __init__(self, datapath, num_classes=9):
        self.data_path = datapath
        self.label_path_train = os.path.join(datapath, "train", "labels")
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

        print(f"Found {len(self.label_paths)} training labels.")
        
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
        """
        ids, counts = np.unique(Y, return_counts=True)
        n_pixels = Y.size

        p_class = np.zeros(n_classes, dtype=np.float64)
        p_class[ids] = counts / n_pixels

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

        class_names = ["background", "car", "person", "bike", "curve", "car stop", "guardrail", "color cone", "bump"]

        for i, name in enumerate(class_names):
            config["ClassWeights"][name] = str(self.weights[i])

        config["ClassWeights"]["num_images"] = str(len(self.label_paths))

        with open(weight_file, "w") as f:
            config.write(f)

        print(f"Saved MFNet weights to: {weight_file}")

    def load_class_weights(self, weight_file):
        config = configparser.ConfigParser()
        config.read(weight_file)

        class_names = ["background", "car", "person", "bike", "curve", "car stop", "guardrail", "color cone", "bump"]

        weights = np.zeros(self.num_classes)
        for i, name in enumerate(class_names):
            weights[i] = float(config["ClassWeights"][name])

        num_images = int(config["ClassWeights"]["num_images"])
        return weights, num_images


def compute_mfnet_weights(path, save=True):
    cw = MFNetClassWeights(path, num_classes=9)
    weights = cw.process_labels()

    if save:
        out_file = os.path.join(path, "mfnet_weights.ini")
        cw.save_class_weights(out_file)

    return weights


if __name__ == "__main__":
    # Example:
    mfnet_path = "/home/ancuti/expendables/VesperMUSE/data/MFNet/"
    compute_mfnet_weights(mfnet_path, save=True)