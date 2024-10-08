# Accurate Spatial Gene Expression Prediction by Integrating Multi-Resolution Features 

> Accurate Spatial Gene Expression Prediction by integrating Multi-resolution features (accepted to CVPR 2024) \
Youngmin Chung, Ji Hun Ha, Kyeong Chan Im, Joo Sang Lee<sup>*

<img src="./figures/TRIPLEX_main.jpg" title="TRIPLEX"/>

## TODO
- [x] Add code to automatically download the ResNet18 weight
- [x] Add code for inference
- [ ] Add code to preprocess WSIs in svs or tif format

## Installation
- Python 3.9.19
- Install pytorch
```bash
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```
- Install the remaining required packages
```bash
pip install -r requirements.txt
```

## Data Preparation and Model Weights
### Step 1: Download Preprocessed Data
Begin by downloading the preprocessed data [here](https://drive.google.com/drive/folders/13oJqeoU5_QPy4_yeZ4eK694AGoBuQjop?usp=drive_link). \
Save the downloaded TRIPLEX.zip file into the ./data directory within your project workspace.

### Step 2: Unzip the Data
After downloading, unzip the TRIPLEX.zip file using the following command:
```bash
unzip ./data/TRIPLEX.zip -d ./data
```
This will extract the data into four subdirectories within the ./data folder, namely her2ts, skin, stnet, and test.

### Step 3: Extract features of slide images
TRIPLEX requires pre-extracted features from WSIs. Run following commands to extract features using pre-trained ResNet18.  
- Cross validation
```python
# BC1 dataset
python preprocess/extract_features.py --config her2st/TRIPLEX --mode internal --extract_mode target
python preprocess/extract_features.py --config her2st/TRIPLEX --mode internal --extract_mode neighbor
# BC2 dataset
python preprocess/extract_features.py --config stnet/TRIPLEX --mode internal --extract_mode target
python preprocess/extract_features.py --config stnet/TRIPLEX --mode internal --extract_mode neighbor
# SCC dataset
python preprocess/extract_features.py --config skin/TRIPLEX --mode internal --extract_mode target
python preprocess/extract_features.py --config skin/TRIPLEX --mode internal --extract_mode neighbor
```

- External test
```python
# 10x Visium-1
python preprocess/extract_features.py --test_name 10x_breast_ff1 --mode external --extract_mode target 
python preprocess/extract_features.py --test_name 10x_breast_ff1 --mode external --extract_mode neighbor
# 10x Visium-2
python preprocess/extract_features.py --test_name 10x_breast_ff2 --mode external --extract_mode target 
python preprocess/extract_features.py --test_name 10x_breast_ff2 --mode external --extract_mode neighbor
# 10x Visium-3
python preprocess/extract_features.py --test_name 10x_breast_ff3 --mode external --extract_mode target 
python preprocess/extract_features.py --test_name 10x_breast_ff3 --mode external --extract_mode neighbor
```

### Directory Structure
After completing the above steps, your project directory should follow this structure: 
```bash
# Directory structure for HER2ST
  .
  ├── data
  │   ├── her2st
  │   │   ├── ST-cnts
  │   │   ├── ST-imgs
  │   │   ├── ST-spotfiles
  │   │   ├── gt_features_224
  │   │   └── n_features_5_224
  └── weights/tenpercent_resnet18.ckpt

```


## Usage
### Training and Testing
- BC1 dataset
```python
# Train
python main.py --config her2st/TRIPLEX --mode cv
# Test
python main.py --config her2st/TRIPLEX --mode test --fold [num_fold] --model_path [path/model/weight]
```

- BC2 dataset
```python
# Train
python main.py --config stnet/TRIPLEX --mode cv
# Test
python main.py --config stnet/TRIPLEX --mode test --fold [num_fold] --model_path [path/model/weight]
```

- SCC dataset
```python
# Train
python main.py --config skin/TRIPLEX --mode cv
# Test
python main.py --config skin/TRIPLEX --mode test --fold [num_fold] --model_path [path/model/weight]
```

Training results will be saved in *./logs*

- Independent test

```python
# 10x Visium-1
python main.py --config skin/TRIPLEX --mode external_test --test_name 10x_breast_ff1 --model_path [path/model/weight]
# 10x Visium-2
python main.py --config skin/TRIPLEX --mode external_test --test_name 10x_breast_ff2 --model_path [path/model/weight]
# 10x Visium-3
python main.py --config skin/TRIPLEX --mode external_test --test_name 10x_breast_ff3 --model_path [path/model/weight]
```

## Acknowledgements
- Code for data processing is based on [HisToGene](https://github.com/maxpmx/HisToGene)
- Code for various Transformer architectures was adapted from [vit-pytorch](https://github.com/lucidrains/vit-pytorch)
- Code for position encoding generator was adapted via making modifications to [TransMIL](https://github.com/szc19990412/TransMIL)
- If you found our work useful in your research, please consider citing our works(s) at:

```
@inproceedings{chung2024accurate,
  title={Accurate Spatial Gene Expression Prediction by integrating Multi-resolution features},
  author={Chung, Youngmin and Ha, Ji Hun and Im, Kyeong Chan and Lee, Joo Sang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11591--11600},
  year={2024}
}
```
