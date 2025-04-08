import os
import random
import yaml
from addict import Dict
from pathlib import Path

import numpy as np
from scipy import sparse
import h5py
import pandas as pd
import scanpy as sc
import torch
from torchvision import transforms
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import wandb

from hest import ( STReader, 
                VisiumReader, 
                VisiumHDReader, 
                XeniumReader )
from hest.HESTData import read_HESTData


def load_st(path, platform):
    assert platform in ['st', 'visium', 'visium-hd', 'xenium'], "platform must be one of ['st', 'visium', 'visium-hd', 'xenium']"
    
    if platform == 'st':    
        st = STReader().auto_read(path)
        
    if platform == 'visium':
        st = VisiumReader().auto_read(path)
        
    if platform == 'visium-hd':
        st = VisiumHDReader().auto_read(path)
        
    if platform == 'xenium':
        # st = XeniumReader().auto_read(path)
        st = read_HESTData(
            adata_path = os.path.join(path, 'aligned_adata.h5ad'),
            img = os.path.join(path, 'aligned_fullres_HE.tif'),
            metrics_path = os.path.join(path, 'metrics.json'),
            # cellvit_path,
            # tissue_contours_path,
            xenium_cell_path = os.path.join(path, 'he_cell_seg.parquet'),
            xenium_nucleus_path = os.path.join(path, 'he_nucleus_seg.parquet'),
            transcripts_path = os.path.join(path, 'aligned_transcripts.parquet')
        )
    return st

def map_values(arr, step_size=256):
    """
    Map NumPy array values to integers such that:
    1. The minimum value is mapped to 0
    2. Values within 256 of each other are mapped to the same integer
    
    Args:
    arr (np.ndarray): Input NumPy array of numeric values
    
    Returns:
    tuple: 
        - NumPy array of mapped integer values 
    """
    if arr.size == 0:
        return np.array([]), {}
    
    # Sort the unique values
    unique_values = np.sort(np.unique(arr))
    
    mapping = {}
    current_key = 0
    
    mapping[unique_values[0]] = 0
    current_value = unique_values[0]

    for i in range(1, len(unique_values)):
        if unique_values[i] - current_value > step_size:
            current_key += 1
            current_value = unique_values[i] 
        
        mapping[unique_values[i]] = current_key
    
    mapped_arr = np.vectorize(mapping.get)(arr)
    
    return mapped_arr

def pxl_to_array(pixel_crds, step_size):
    x_crds = map_values(pixel_crds[:,0], step_size)
    y_crds = map_values(pixel_crds[:,1], step_size)
    dst = np.stack((x_crds, y_crds), axis=1)
    return dst

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def normalize_adata(adata: sc.AnnData, cpm=False, smooth=False) -> sc.AnnData:
    """
    Normalize each spot by total gene counts + Logarithmize each spot
    """
    # print(f"Number of spots before filetering: {adata.shape[0]}")
    # normed_adata.X = normed_adata.X.astype(np.float64)
    # sc.pp.filter_cells(adata, min_counts=100)
    # print(f"Number of spots after filetering: {adata.shape[0]}")
    
    normed_adata = adata.copy()
    
    if cpm:
        # Normalize each spot
        sc.pp.normalize_total(normed_adata, target_sum=1e4)

    # Logarithm of the expression
    sc.pp.log1p(normed_adata)

    if smooth:
        new_X = []
        for index, df_row in normed_adata.obs.iterrows():
            row = int(df_row['array_row'])
            col = int(df_row['array_col'])
            
            neighbors_index = normed_adata.obs[((normed_adata.obs['array_row'] >= row - 1) & (normed_adata.obs['array_row'] <= row + 1)) & \
                ((normed_adata.obs['array_col'] >= col - 1) & (normed_adata.obs['array_col'] <= col + 1))].index
            neighbors = normed_adata[neighbors_index]
            nb_neighbors = len(neighbors)
            
            avg = neighbors.X.sum(0) / nb_neighbors
            new_X.append(avg)
            # normed_normed_adata[i] = avg            
        
        new_X = np.stack(new_X)
        normed_adata.X = new_X
        # if sparse.issparse(adata.X):
        #     adata.X = sparse.csr_matrix(new_X)
        # else:
        #     adata.X = new_X

    return normed_adata

# Load config
def load_config(config_name: str):
    """load config file in Dict

    Args:
        config_name (str): Name of config file. 

    Returns:
        Dict: Dict instance containing configuration.
    """
    config_path = os.path.join('./config', f'{config_name}.yaml')

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader = yaml.FullLoader)
    
    return Dict(config)

# Load loggers
def load_loggers(cfg: Dict):
    """Return Logger instance for Trainer in List.

    Args:
        cfg (Dict): Dict containing configuration.

    Returns:
        List: _description_
    """
    
    log_path = cfg.GENERAL.log_path
    current_time = cfg.GENERAL.timestamp
    
    
    Path(log_path).mkdir(exist_ok=True, parents=True)
    log_name = str(Path(cfg.config).parent)
    version_name = Path(cfg.config).name
    # now = datetime.now()
    # current_time = now.strftime("%D-%H-%M").replace("/", "-")
    cfg.log_path = f"{log_path}/{log_name}/{version_name}/{current_time}/fold{cfg.DATA.fold}"
    print(f'---->Log dir: {cfg.log_path}')
    

    # #---->Wandb
    # log_path = os.path.abspath(cfg.log_path)
    os.environ["WANDB_DIR"] = cfg.log_path
    # os.environ["WANDB_DISABLE_SERVICE"] = "True"
    # os.environ["WANDB_DIR"] = os.path.expanduser("~/.wandb")
    os.makedirs(f'{cfg.log_path}/wandb', exist_ok=True)
    wandb_logger = pl_loggers.WandbLogger(save_dir=cfg.log_path,
                                        name=f'{log_name}-{version_name}-{current_time}-fold{cfg.DATA.fold}', 
                                        project='ST_prediction')
    
    # wandb.init(
    #     dir=cfg.log_path,  # 로그 저장 경로 지정
    #     name=f'{log_name}-{version_name}-{current_time}-fold{cfg.DATA.fold}',
    #     project='ST_prediction'
    # )
    # wandb_logger = pl_loggers.WandbLogger(experiment=wandb.run)
    # tb_logger = pl_loggers.TensorBoardLogger(log_path+log_name,
    #                                         name = f"{version_name}/{current_time}_{cfg.exp_id}", version = f'fold{cfg.Data.fold}',
    #                                         log_graph = True, default_hp_metric = False)
    #---->CSV
    csv_logger = pl_loggers.CSVLogger(f"{log_path}/{log_name}",
                                    name = f"{version_name}/{current_time}", version = f'fold{cfg.DATA.fold}', )
    
    # log_path = os.path.join(cfg.GENERAL.log_path, cfg.GENERAL.timestamp)

    # tb_logger = TensorBoardLogger(
    #     log_path,
    #     name = cfg.GENERAL.log_name)

    # csv_logger = CSVLogger(
    #     log_path,
    #     name = cfg.GENERAL.log_name)
    
    loggers = [wandb_logger, csv_logger]
    
    return loggers

# load Callback
def load_callbacks(cfg: Dict):
    """Return Early stopping and Checkpoint Callbacks. 

    Args:
        cfg (Dict): Dict containing configuration.

    Returns:
        List: Return List containing the Callbacks.
    """
    
    Mycallbacks = []
    
    target = 'val_target'
    patience = cfg.TRAINING.early_stopping.patience
    mode = cfg.TRAINING.mode
    
    early_stop_callback = EarlyStopping(
        monitor=target,
        min_delta=0.00,
        patience=patience,
        verbose=True,
        mode=mode
    )
    Mycallbacks.append(early_stop_callback)
    log_name = '{epoch:02d}-{val_target:.4f}'
    checkpoint_callback = ModelCheckpoint(monitor = target,
                                    dirpath = cfg.log_path,
                                    filename = log_name,
                                    verbose = True,
                                    save_last = False,
                                    save_top_k = 1,
                                    mode = mode,
                                    save_weights_only = True)
    Mycallbacks.append(checkpoint_callback)
        
    return Mycallbacks

def save_hdf5(output_fpath, 
                  asset_dict, 
                  attr_dict= None, 
                  mode='a', 
                  auto_chunk = True,
                  chunk_size = None):
    """
    output_fpath: str, path to save h5 file
    asset_dict: dict, dictionary of key, val to save
    attr_dict: dict, dictionary of key: {k,v} to save as attributes for each key
    mode: str, mode to open h5 file
    auto_chunk: bool, whether to use auto chunking
    chunk_size: if auto_chunk is False, specify chunk size
    """
    with h5py.File(output_fpath, mode) as f:
        for key, val in asset_dict.items():
            data_shape = val.shape
            if len(data_shape) == 1:
                val = np.expand_dims(val, axis=1)
                data_shape = val.shape

            if key not in f: # if key does not exist, create dataset
                data_type = val.dtype
                if data_type == np.object_: 
                    data_type = h5py.string_dtype(encoding='utf-8')
                if auto_chunk:
                    chunks = True # let h5py decide chunk size
                else:
                    chunks = (chunk_size,) + data_shape[1:]
                try:
                    dset = f.create_dataset(key, 
                                            shape=data_shape, 
                                            chunks=chunks,
                                            maxshape=(None,) + data_shape[1:],
                                            dtype=data_type)
                    ### Save attribute dictionary
                    if attr_dict is not None:
                        if key in attr_dict.keys():
                            for attr_key, attr_val in attr_dict[key].items():
                                dset.attrs[attr_key] = attr_val
                    dset[:] = val
                except:
                    print(f"Error encoding {key} of dtype {data_type} into hdf5")
                
            else:
                dset = f[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                assert dset.dtype == val.dtype
                dset[-data_shape[0]:] = val
        
        # if attr_dict is not None:
        #     for key, attr in attr_dict.items():
        #         if (key in asset_dict.keys()) and (len(asset_dict[key].attrs.keys())==0):
        #             for attr_key, attr_val in attr.items():
        #                 dset[key].attrs[attr_key] = attr_val
                
    return output_fpath



def get_transforms(mean, std, target_img_size = -1, center_crop = False, transform_type = 'eval'):
    trsforms = []
    
    # Apply specific transformation based on transform_type
    if transform_type == 'hori':
        # Horizontal flip
        trsforms.append(transforms.RandomHorizontalFlip(p=1.0))
    elif transform_type == 'vert':
        # Vertical flip
        trsforms.append(transforms.RandomVerticalFlip(p=1.0))
    elif transform_type == 'rot_90':
        # 90-degree rotation
        trsforms.append(transforms.RandomRotation((90, 90)))
    elif transform_type == 'rot_180':
        # 180-degree rotation
        trsforms.append(transforms.RandomRotation((180, 180)))
    elif transform_type == 'rot_270':
        # 270-degree rotation
        trsforms.append(transforms.RandomRotation((270, 270)))
    elif transform_type == 'tp':
        # Transpose (90-degree rotation + horizontal flip)
        class Transpose(object):
            def __call__(self, img):
                return transforms.functional.hflip(transforms.functional.rotate(img, 90))
        trsforms.append(Transpose())
    elif transform_type == 'tv':
        # Transverse (90-degree rotation + vertical flip)
        class Transverse(object):
            def __call__(self, img):
                return transforms.functional.vflip(transforms.functional.rotate(img, 90))
        trsforms.append(Transverse())
        
    elif transform_type == 'eval':
        # Default 'eval' mode has no augmentations
        pass
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")
    
    if target_img_size > 0:
        trsforms.append(transforms.Resize(target_img_size))
    if center_crop:
        assert target_img_size > 0, "target_img_size must be set if center_crop is True"
        trsforms.append(transforms.CenterCrop(target_img_size))
        
    trsforms.append(transforms.ToTensor())
    if mean is not None and std is not None:
        trsforms.append(transforms.Normalize(mean, std))
    trsforms = transforms.Compose(trsforms)

    return trsforms



def add_augmentation_to_transform(existing_transform, transform_type='eval'):
    """
    Adds the specified augmentation to an existing transform pipeline.
    
    Args:
        existing_transform (transforms.Compose): Existing transformation pipeline
        transform_type (str): Type of augmentation to add ('hori', 'vert', 'rot_90', 
                            'rot_180', 'rot_270', 'tp', 'tv', or 'eval')
    
    Returns:
        transforms.Compose: New transformation pipeline with added augmentation
    """
    from torchvision import transforms
    
    # Create augmentation based on transform_type
    augmentation = None
    if transform_type == 'hori':
        # Horizontal flip
        augmentation = transforms.RandomHorizontalFlip(p=1.0)
    elif transform_type == 'vert':
        # Vertical flip
        augmentation = transforms.RandomVerticalFlip(p=1.0)
    elif transform_type == 'rot_90':
        # 90-degree rotation
        augmentation = transforms.RandomRotation((90, 90))
    elif transform_type == 'rot_180':
        # 180-degree rotation
        augmentation = transforms.RandomRotation((180, 180))
    elif transform_type == 'rot_270':
        # 270-degree rotation
        augmentation = transforms.RandomRotation((270, 270))
    elif transform_type == 'tp':
        # Transpose (90-degree rotation + horizontal flip)
        class Transpose(object):
            def __call__(self, img):
                return transforms.functional.hflip(transforms.functional.rotate(img, 90))
        augmentation = Transpose()
    elif transform_type == 'tv':
        # Transverse (90-degree rotation + vertical flip)
        class Transverse(object):
            def __call__(self, img):
                return transforms.functional.vflip(transforms.functional.rotate(img, 90))
        augmentation = Transverse()
    elif transform_type == 'eval':
        # No augmentation in eval mode
        return existing_transform
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")
    
    # Extract transforms from the existing pipeline
    transform_list = list(existing_transform.transforms)
    
    # Insert the augmentation at the beginning of the pipeline
    transform_list.insert(0, augmentation)
    
    # Return new transform pipeline
    return transforms.Compose(transform_list)