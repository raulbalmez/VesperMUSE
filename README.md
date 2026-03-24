# VesperMUSE: Multi-Modal Fusion Vision Transformer for Image Semantic Segmentation Under Diverse Light Conditions

VesperMUSE is a multi-modal Vision Transformer designed to perform robust image semantic segmentation across a wide variety of lighting conditions.


## Installation

**Prerequisite:** Ensure you have [Anaconda or Miniconda](https://docs.anaconda.com/miniconda/) installed on your system.

1. Clone this repository:
```bash
git clone https://github.com/raulbalmez/VesperMuse.git
```

2. Create environment (multiple ways of creating the environment exist using different version of Python, but we recommend the following as it was the one used for development)

```bash
cd VesperMuse
conda create -n vespermuse python=3.9.25 -y
conda activate vespermuse
```


3. Install the requirements
```bash
pip install -r requirements.txt
```
Note: There might be an issue with the torch installation. If that is the case please run this first and after that install the requirements as indicated above:
```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

4. Usually, the variable PYTHONPATH must be set for a smooth run and package finding.
```bash
export PYTHONPATH=$PYTHONPATH:<path/to/VesperMuse/basicsr>
```

## Training

Model training is controlled via configuration files located in the Options/ directory. To begin training, run the script below and specify the configuration that matches your desired setup:

```bash
python basicsr/train.py --opt <path/to/option.yml>
```


## Inference

To evaluate the models, you must provide the original training configuration file along with the corresponding saved model weights (typically generated and saved in the experiments/ directory).

1. Segmentation

_Note: You must also specify the number of target classes based on your chosen dataset (e.g., 5 for PST900 or 10 for CART)._

```bash
python Enhancement/test_from_dataset_SEG.py --opt <path/to/option.yml> --weights <path/to/weights.pth> --num_classes <NUM_CLASSES> --tta
```

2. Low-Light Image Enhancement
   
```bash
python Enhancement/test_from_dataset_LLIE.py --opt <path/to/option.yml> --weights <path/to/weights.pth> --tta
```

## Options (configuration files)

_Note: The network can be trained in multiple ways using multiple configuration files (eg. using LLIE or not, checkpoints for different LLIE usages and so on)._

All the configurations used for development have been provided. These can be found in the _Options/_ folder and stand as a very comprehensive guideline to create more ways in which VesperMUSE can be run. Before training, read through each field, as they are self-explanatory and feel free to modify the fields that best suit your purpose.


## Weights and data

Access the datasets (which will be put under the data/ folder) as well as the weights.

https://drive.google.com/drive/folders/1VZwq-X9LnLF8Hghc-vgy1ppPXGaIgtce?usp=drive_link

