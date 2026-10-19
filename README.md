<div align="center">

# VesperMUSE

### Multi-Modal Fusion Vision Transformer for Image Semantic Segmentation Under Diverse Light Conditions

[![arXiv](https://img.shields.io/badge/arXiv-2610.21990-b31b1b.svg?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2610.21990)
[![Python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-EE4C2C.svg?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Weights & Data](https://img.shields.io/badge/Weights%20%26%20Data-Google%20Drive-4285F4.svg?style=flat-square&logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1VZwq-X9LnLF8Hghc-vgy1ppPXGaIgtce?usp=drive_link)

[**Paper**](https://arxiv.org/abs/2610.21990) · [**Weights & Data**](https://drive.google.com/drive/folders/1VZwq-X9LnLF8Hghc-vgy1ppPXGaIgtce?usp=drive_link) · [**Results**](#-results) · [**Citation**](#-citation)

</div>

---

**VesperMUSE** is a multi-modal Vision Transformer for robust semantic segmentation across a wide range of lighting conditions, from well-lit scenes to severe low light. Alongside segmentation, it provides a lightweight low-light image enhancement (LLIE) branch, enabling joint enhancement-and-segmentation pipelines.

## 📌 Highlights

- **State of the art on CART** — **76.72% mIoU**, surpassing the previous best (CM-SSM, 74.60%) by **+2.12 mIoU**, with the top score on 8 of 10 classes.
- **Competitive on PST900** — **85.59% mIoU**, with the best IoU on *Background*, *Hand-Drill* and *Fire-Extinguisher*.
- **Efficient low-light enhancement** — the LLIE branch reaches **27.47 dB / 27.74 dB PSNR** on PST900 LL / CART LL with only **0.45M parameters**, ~4× fewer than the strongest baseline while staying within 0.11 dB.
- **Synthetic low-light benchmarks** — PST900 LL and CART LL, paired low-light adaptations whose statistics (SNR ≈ 10.7 dB, gamma ≈ 1.00) closely match real low-light datasets such as LOL.

## 📊 Results

### Semantic Segmentation

| Method | Backbone | PST900 mIoU (%) ↑ | CART mIoU (%) ↑ |
| :--- | :---: | :---: | :---: |
| LASNet | ResNet-152 | 84.40 | – |
| CMX | Mit-B2 | – | 74.00 |
| CM-SSM | EfficientViT-B1 | **85.90** | 74.60 |
| **VesperMUSE (ours)** | ResNet50 / ResNet18 | 85.59 | **76.72** |

<details>
<summary><b>Per-class highlights</b></summary>

| Dataset | Class | VesperMUSE IoU (%) | Previous best (%) |
| :--- | :--- | :---: | :---: |
| PST900 | Hand-Drill | **82.82** | 81.01 (SA-Gate) |
| PST900 | Fire-Extinguisher | **87.76** | 85.70 (CM-SSM) |
| CART | Bare Ground | **87.06** | 82.60 (CM-SSM) |
| CART | Person | **34.26** | 31.50 (MCNet-T) |
| CART | Vehicles | **63.14** | 62.00 (CM-SSM) |

</details>

### Low-Light Image Enhancement

| Method | PST900 LL PSNR / SSIM ↑ | CART LL PSNR / SSIM ↑ | Params (M) ↓ |
| :--- | :---: | :---: | :---: |
| Retinexformer | 26.60 / 0.850 | 27.11 / 0.847 | 1.61 |
| ISALux | **27.58 / 0.874** | **27.81 / 0.873** | 1.85 |
| **VesperMUSE LLIE (ours)** | 27.47 / 0.870 | 27.74 / 0.858 | **0.454** |

Full comparison tables are available in the [paper](https://arxiv.org/abs/2610.21990).

## ⚙️ Installation

**Prerequisite:** [Anaconda or Miniconda](https://docs.anaconda.com/miniconda/).

**1. Clone the repository**

```bash
git clone https://github.com/raulbalmez/VesperMUSE.git
cd VesperMUSE
```

**2. Create the environment** (Python 3.9.25 was used for development and is recommended)

```bash
conda create -n vespermuse python=3.9.25 -y
conda activate vespermuse
```

**3. Install the requirements**

```bash
pip install -r requirements.txt
```

> [!NOTE]
> If the PyTorch installation fails, install it first and then re-run the command above:
> ```bash
> pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
> ```

**4. Set `PYTHONPATH`** so the packages are found correctly

```bash
export PYTHONPATH=$PYTHONPATH:<path/to/VesperMUSE/basicsr>
```

## 📂 Weights and Data

Download the datasets and pretrained weights from **[Google Drive](https://drive.google.com/drive/folders/1VZwq-X9LnLF8Hghc-vgy1ppPXGaIgtce?usp=drive_link)**. Place the datasets under the `data/` folder.

## 🏋️ Training

Training is driven by configuration files in the `Options/` directory. Pick the configuration that matches your setup:

```bash
python basicsr/train.py --opt <path/to/option.yml>
```

## 🔍 Inference

Evaluation requires the original training configuration and the corresponding model weights (saved by default under `experiments/`).

**Semantic segmentation** — set `--num_classes` according to the dataset (`5` for PST900, `10` for CART):

```bash
python Enhancement/test_from_dataset_SEG.py \
    --opt <path/to/option.yml> \
    --weights <path/to/weights.pth> \
    --num_classes <NUM_CLASSES> \
    --tta
```

**Low-light image enhancement:**

```bash
python Enhancement/test_from_dataset_LLIE.py \
    --opt <path/to/option.yml> \
    --weights <path/to/weights.pth> \
    --tta
```

## 🔧 Configuration

VesperMUSE can be trained in several ways (with or without LLIE, from different LLIE checkpoints, and more). All configurations used during development are provided in `Options/` and serve as a comprehensive starting point for new setups. The fields are self-explanatory — read through them before training and adapt them to your needs.

## 📄 Citation

If you find VesperMUSE useful in your research, please consider citing:

```bibtex
@article{vespermuse2026,
  title   = {VesperMUSE: Multi-Modal Fusion Vision Transformer for Image Semantic Segmentation Under Diverse Light Conditions},
  author  = {Balmez, Raul},
  journal = {arXiv preprint arXiv:2610.21990},
  year    = {2026}
}
```

## Acknowledgements

This codebase builds on [BasicSR](https://github.com/XPixelGroup/BasicSR). Thanks to the authors of [PST900](https://github.com/ShreyasSkandanS/pst900_thermal_rgb) and [CART](https://arxiv.org/abs/2404.19385) for making their datasets publicly available.

<div align="center">

If you find this work useful, please consider starring the repository.

</div>
