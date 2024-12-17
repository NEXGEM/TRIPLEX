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
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from hest import ( STReader, 
                VisiumReader, 
                VisiumHDReader, 
                XeniumReader )


def load_st(path, platform):
    assert platform in ['st', 'visium', 'visium-hd', 'xenium'], "platform must be one of ['st', 'visium', 'visium-hd', 'xenium']"
    
    if platform == 'st':    
        st = STReader().auto_read(path)
        
    if platform == 'visium':
        st = VisiumReader().auto_read(path)
        
    if platform == 'visium-hd':
        st = VisiumHDReader().auto_read(path)
        
    if platform == 'xenium':
        st = XeniumReader().auto_read(path)
        
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
        sc.pp.normalize_total(normed_adata)

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
    wandb_logger = pl_loggers.WandbLogger(save_dir=cfg.log_path,
                                        name=f'{log_name}-{version_name}-{current_time}-fold{cfg.DATA.fold}', 
                                        project='ST_prediction')
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