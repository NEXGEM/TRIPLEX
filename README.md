# TRIPLEX 🧬

Accurate Spatial Gene Expression Prediction by integrating Multi-resolution features, CVPR 2024. [[arXiv]](https://arxiv.org/abs/2403.07592) \
Youngmin Chung, Ji Hun Ha, Kyeong Chan Im, Joo Sang Lee<sup>*

[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyTorch Lightning](https://img.shields.io/badge/pytorch-lightning-blue.svg?logo=PyTorch%20Lightning)](https://lightning.ai/docs/pytorch/stable/)

<img src="./figures/TRIPLEX_main.jpg" title="TRIPLEX"/>

It is now integrated with [HEST](https://github.com/mahmoodlab/HEST) and [CLAM](https://github.com/mahmoodlab/CLAM) for data preparation.

## 🔄 Recent Updates
- **April 2025**: 
  - Added clamping to outputs to ensure non-negative gene expression values
  - Modified APEG to align with its original implementation intent 
  - Improved stability of training and inference
  - Restructured configuration system with default and model-specific config files
    - Added `default.yaml` with common settings
    - Created modular configs that inherit and override default settings
  - Added flexible position encoding options
    - APEG
    - Optional MLP-based positional encoding
    - Ability to disable positional encoding

- **May 2025**: 
  - Additional position encoding options
    - SpatialFormer (Xiao, Han, et al. "SpatialFormer: Towards Generalizable Vision Transformers with Explicit Spatial Understanding." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024.)

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Preprocessing](#preprocessing)
  - [Training](#-training)
  - [Evaluation](#-evaluation)
  - [Inference](#-inference)
- [Configuration](#configuration)
- [Citation](#Citation)

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- PyTorch Lightning 2.0+
- CUDA-enabled GPU (recommended)

TRIPLEX is tested on Python 3.11, PyTorch 2.4.1, PyTorch Lightning 2.4.0, and CUDA 12.1.

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

Install [FlashAttention](https://github.com/Dao-AILab/flash-attention)

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
├── figures/                # Figures
├── src/                    # Source code
│   ├── dataset/            # Dataset loading and preprocessing modules
│   ├── model/              # Model architectures
│   ├── preprocess/         # Codes for preprocessing data
│   ├── experiment/         # Codes for organizing experiment results
│   ├── main.py             # Main script for training, evaluation, and inference
│   ├── utils.py            # Utility functions
├── script/                 # Example scripts for runs 
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
```

---

## Usage

### Preprocessing

Before training or inference, raw data must be preprocessed. \
Due to the complexity of data preprocessing, we've updated the data pipeline module. \
See the following tutorial: [data_processing.ipynb](tutorials/data_processing.ipynb)

> [!NOTE]
>
> **Reproducing our experiments:**
> The ST datasets we used in our experiments are already included in HEST data.
> Use 'hest' mode in data pipeline to pre-process ST datasets.


### 📈 Training

To train the model using cross-validation, run the following command:

```bash
python src/main.py --config_name=<config_path> --mode=cv --gpu=1
```

Replace `<config_path>` with the path to your configuration file.

### 📊 Evaluation

To evaluate the model, run the following command:

```bash
python src/main.py --config_name=<config_path> --mode=eval --gpu=1 --timestamp=<timestamp_of_log_dir>
```

The most recent folder inside the log directory will be used for evaluation. The file `pcc_rank.npy` will be saved in the output directory.

To identify highly predictive genes (HPGs), use the following command:

```bash
python src/experiment/get_HPG.py --dataset=<dataset_name> --model=<model_name>
```

The file `idx_top.npy` will be saved in the output directory.

To evaluate the model including HPGs, run the evaluation command again:

```bash
python src/main.py --config_name=<config_path> --mode=eval --gpu=1 --timestamp=<timestamp_of_log_dir>
```


### 🔍 Inference

To run inference:

```bash
python src/main.py --config_name=<config_path> --mode=inference --gpu=1 --model_path=<model_checkpoint_path>
```

Replace `<model_checkpoint_path>` with the path to your trained model checkpoint.

---

## Configuration

TRIPLEX now uses a hierarchical configuration system with YAML files located in the `config/` directory:

### Default Configuration

The `config/default.yaml` file contains common settings shared across all experiments:

```yaml
GENERAL:
  seed: 2021
  log_path: ./logs
  
TRAINING:
  num_k: 8
  learning_rate: 1.0e-4
  num_epochs: 200
  monitor: PearsonCorrCoef
  mode: max
  early_stopping:
    patience: 20
  lr_scheduler:
    patience: 10
    factor: 0.1

DATA:
  data_dir: input/ST/andersson
  output_dir: output/pred/ST/andersson
  gene_type: 'mean'
  num_genes: 1000
  num_outputs: 250
  cpm: True
  smooth: True
  model_name: 'cigar'
  
  train_dataloader:
        batch_size: 128
        num_workers: 4
        pin_memory: False
        shuffle: True

  test_dataloader:
      batch_size: 1
      num_workers: 4
      pin_memory: False
      shuffle: False
```

### Model-Specific Configuration

Model-specific configurations inherit from the default and only need to specify differences:

```yaml
MODEL:
  model_name: TRIPLEX 
  num_genes: 250
  emb_dim: 512
  depth1: 1
  depth2: 3
  depth3: 3
  num_heads1: 4
  num_heads2: 16
  num_heads3: 16
  mlp_ratio1: 4
  mlp_ratio2: 4
  mlp_ratio3: 1
  dropout1: 0.2
  dropout2: 0.1
  dropout3: 0.3
  kernel_size: 3

DATA:
  dataset_name: TriDataset
```

Modify these files as needed for your experiments.

---

## Citation

```
@inproceedings{chung2024accurate,
  title={Accurate Spatial Gene Expression Prediction by integrating Multi-resolution features},
  author={Chung, Youngmin and Ha, Ji Hun and Im, Kyeong Chan and Lee, Joo Sang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11591--11600},
  year={2024}
}
```

---



