# TRIPLEX

TRIPLEX is a deep learning framework designed for predicting spatial transcriptomics from histology images by integrating multi-resolution features. It is now integrated with HEST for enhanced functionality.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Preprocessing](#preprocessing)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- PyTorch Lightning 2.0+
- CUDA-enabled GPU (recommended)

TRIPLEX is tested on Python 3.11, PyTorch 2.4.1, PyTorch Lightning 2.4.0, and CUDA 12.1.
Additional dependencies are listed in the `requirements.txt` file. To install them, run:

```bash
pip install -r requirements.txt
```

---

## Installation

Clone this repository:

```bash
git clone https://github.com/NEXGEM/TRIPLEX.git
cd triplex
```

Create a conda environment:

```bash
conda create -n TRIPLEX python=3.11
```

Install Pytorch 

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

Install HEST

- Dependencies (CUDA-related Python packages)

```bash
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu12==24.6.0 \
    dask-cudf-cu12==24.6.0 \
    cucim-cu12==24.6.0 \
    raft-dask-cu12==24.6.0
```

- HEST

```bash
git clone https://github.com/mahmoodlab/HEST.git 
cd HEST 
pip install -e .
```

Instann FlashAttention

```bash
pip install flash-attn --no-build-isolation
```

Install remaining dependencies:

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
.
├── config/                 # Configuration files for experiments
├── docker/                 # Docker-related setup files
├── figures/                # Figures for results and documentation
├── src/                    # Source code
│   ├── datasets/           # Dataset loading and preprocessing modules
│   ├── models/             # Model architectures
│   ├── preprocess/         # Scripts for preprocessing data
│   ├── __init__.py         # Package initialization
│   ├── main.py             # Main script for training and inference
│   ├── utils.py            # Utility functions
├── 01-preprocess_for_training.sh  # Preprocessing script for training
├── 02-preprocess_for_inference.sh # Preprocessing script for inference
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── run_extract_features.txt# Example feature extraction commands
```

---

## Usage

### Preprocessing

Before training or inference, raw data must be preprocessed. Modify the paths in the respective shell scripts and run them:

#### For Training:
```bash
bash 01-preprocess_for_training.sh
```

#### For Inference:
```bash
bash 02-preprocess_for_inference.sh
```

### Training

To train the model using cross-validation, run the following command:

```bash
python src/main.py --config_name=<config_path> --mode=cv --gpu=0
```

Replace `<config_path>` with the path to your configuration file.

### Evaluation

To evaluate the model:

```bash
python src/main.py --config_name=<config_path> --mode=eval --gpu=0 --model_path=<model_checkpoint_path>
```

Replace `<model_checkpoint_path>` with the path to your trained model checkpoint.

### Inference

To run inference:

```bash
python src/main.py --config_name=<config_path> --mode=inference --gpu=0 --model_path=<model_checkpoint_path>
```

---

## Configuration

Configurations are managed using YAML files located in the `config/` directory. Each configuration file specifies parameters for the dataset, model, training, and evaluation. Example configuration parameters include:

```yaml
GENERAL:
  seed: 2021
  log_path: ./logs
  
TRAINING:
  num_k: 5
  batch_size: 128
  loss: MSE
  optimizer: adam
  learning_rate: 1.0e-4
  num_epochs: 200
  early_stopping:
    monitor: val_MeanSquaredError
    patience: 20
    mode: min
  lr_scheduler:
    monitor: val_MeanSquaredError
    patience: 5
    factor: 0.1
    mode: min
  
MODEL:
  model_name: TRIPLEX 
  num_outputs: 1000
  emb_dim: 1024
  depth1: 1
  depth2: 5
  depth3: 4
  num_heads1: 8
  num_heads2: 8
  num_heads3: 8
  mlp_ratio1: 4
  mlp_ratio2: 4
  mlp_ratio3: 4
  dropout1: 0.4
  dropout2: 0.3
  dropout3: 0.3
  kernel_size: 3
  learning_rate: 0.0001

DATA:
  data_dir: input/path/to/data
  dataset_name: tri_dataset
  
  train_dataloader:
        batch_size: 128 
        num_workers: 4
        pin_memory: True
        shuffle: True

  test_dataloader:
      batch_size: 1
      num_workers: 4
      pin_memory: True
      shuffle: False
```

Modify these files as needed for your experiments.

---

## Citation

@inproceedings{chung2024accurate,
  title={Accurate Spatial Gene Expression Prediction by integrating Multi-resolution features},
  author={Chung, Youngmin and Ha, Ji Hun and Im, Kyeong Chan and Lee, Joo Sang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11591--11600},
  year={2024}
}

---



