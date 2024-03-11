# Accurate Spatial Gene Expression Prediction by integrating Multi-resolution features 

> Accurate Spatial Gene Expression Prediction by integrating Multi-resolution features (accepted in CVPR 2024) \
Youngmin Chung, Ji Hun Ha, Kyeong Chan Im, Joo Sang Lee<sup>*

<img src="./figures/TRIPLEX_main.jpg" title="TRIPLEX"/>

## TODO
 - [x] Add implementation code for TRIPLEX
 - [x] Complete requirements.txt
 
## Installation
- Python 3.8.16

```bash
pip install -r requirements.txt
```

## Data Preparation and Model Weights
### Step 1: Download Preprocessed Data
- Begin by downloading the preprocessed data [here](https://drive.google.com/drive/folders/13oJqeoU5_QPy4_yeZ4eK694AGoBuQjop?usp=drive_link).
- Save the downloaded TRIPLEX.zip file into the ./data directory within your project workspace.

### Step 2: Unzip the Data
- After downloading, unzip the TRIPLEX.zip file using the following command:
```bash
unzip ./data/TRIPLEX.zip -d ./data
```
This will extract the data into four subdirectories within the ./data folder, namely her2ts, skin, stnet, and test.

### Step 3: Pre-trained Weights
- Ensure that you have the pre-trained weights of ResNet18, as provided by Ciga et al., stored within the ./weights directory of your project workspace. 

### Directory Structure
- After completing the above steps, your project directory should follow this structure: 
```bash
  .
  ├── data
  │   ├── her2st
  │   ├── skin
  │   ├── stnet
  │   └── test
  └── weights/tenpercent_resnet18.ckpt
```

### Step 4: Extract features
- Cross validation
```python
# BC1 dataset
python extract_features.py --config her2st/TRIPLEX --test_mode internal --extract_mode g_target
python extract_features.py --config her2st/TRIPLEX --test_mode internal --extract_mode neighbor
# BC2 dataset
python extract_features.py --config stnet/TRIPLEX --test_mode internal --extract_mode g_target
python extract_features.py --config stnet/TRIPLEX --test_mode internal --extract_mode neighbor
# SCC dataset
python extract_features.py --config skin/TRIPLEX --test_mode internal --extract_mode g_target
python extract_features.py --config skin/TRIPLEX --test_mode internal --extract_mode neighbor
```

- External test
```python
# 10x Visium-1
python extract_features.py --test_name 10x_breast_ff1 --test_mode external --extract_mode g_target 
python extract_features.py --test_name 10x_breast_ff1 --test_mode external --extract_mode neighbor
# 10x Visium-2
python extract_features.py --test_name 10x_breast_ff2 --test_mode external --extract_mode g_target 
python extract_features.py --test_name 10x_breast_ff2 --test_mode external --extract_mode neighbor
# 10x Visium-3
python extract_features.py --test_name 10x_breast_ff3 --test_mode external --extract_mode g_target 
python extract_features.py --test_name 10x_breast_ff3 --test_mode external --extract_mode neighbor
```


## Usage
### Training and Testing

* BC1 dataset
```python
# Train
python main.py --config her2st/TRIPLEX --mode cv
# Test
python main.py --config her2st/TRIPLEX --mode test
```

* BC2 dataset
```python
# Train
python main.py --config stnet/TRIPLEX --mode cv
# Test
python main.py --config stnet/TRIPLEX --mode test
```

* SCC dataset
```python
# Train
python main.py --config skin/TRIPLEX --mode cv
# Test
python main.py --config skin/TRIPLEX --mode test
```

* Independent test

```python
# 10x Visium-1
python main.py --config skin/TRIPLEX --mode ex_test --test_name 10x_breast_ff1
# 10x Visium-2
python main.py --config skin/TRIPLEX --mode ex_test --test_name 10x_breast_ff2
# 10x Visium-3
python main.py --config skin/TRIPLEX --mode ex_test --test_name 10x_breast_ff3
```

## Acknowledgements
- Code for data processing is based on [HisToGene](https://github.com/maxpmx/HisToGene)
- Code for various Transformer architectures was adapted from [vit-pytorch](https://github.com/lucidrains/vit-pytorch)
- Code for position encoding generator was adapted via making modifications to [TransMIL](https://github.com/szc19990412/TransMIL)
- If you found our work useful in your research, please consider citing our works(s) at:

```
@article{chung2024accurate,
  title={Accurate Spatial Gene Expression Prediction by integrating Multi-resolution features },
  author={Youngmin Chung, Ji Hun Ha, Kyeong Chan Im, Joo Sang Lee},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  year={2024}
}
```
