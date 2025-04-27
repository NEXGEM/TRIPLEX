import os
import random
import yaml
from addict import Dict
from pathlib import Path

import numpy as np
from scipy import sparse

import scanpy as sc
import torch

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


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
def load_config(config_path: str):
    """load config file in Dict

    Args:
        config_name (str): Name of config file. 

    Returns:
        Dict: Dict instance containing configuration.
    """
    # config_path = os.path.join(config_dir, f'{config_name}.yaml')

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
    
    log_root_dir = cfg.GENERAL.log_path
    current_time = cfg.GENERAL.timestamp
    log_name = str(Path(cfg.config).parent)
    version_name = Path(cfg.config).name
    
    cfg.log_dir_fold = f"{cfg.log_dir}/fold{cfg.DATA.fold}"
    print(f'---->Log dir: {cfg.log_dir}')
    

    # #---->Wandb
    # log_path = os.path.abspath(cfg.log_path)
    os.environ["WANDB_DIR"] = cfg.log_dir_fold
    # os.environ["WANDB_DISABLE_SERVICE"] = "True"
    # os.environ["WANDB_DIR"] = os.path.expanduser("~/.wandb")
    os.makedirs(f'{cfg.log_dir_fold}/wandb', exist_ok=True)
    wandb_logger = pl_loggers.WandbLogger(save_dir=cfg.log_dir_fold,
                                        name=f'{log_name}-{version_name}-{current_time}-fold{cfg.DATA.fold}', 
                                        project='ST_prediction')
    
    #---->CSV
    csv_logger = pl_loggers.CSVLogger(f"{log_root_dir}/{log_name}",
                                    name = f"{version_name}/{current_time}", version = f'fold{cfg.DATA.fold}', )
    
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
                                    dirpath = cfg.log_dir_fold,
                                    filename = log_name,
                                    verbose = True,
                                    save_last = False,
                                    save_top_k = 1,
                                    mode = mode,
                                    save_weights_only = True)
    Mycallbacks.append(checkpoint_callback)
        
    return Mycallbacks


def load_config_with_default(config_path, base_config_path=None):
    """Load config with base config support."""
    
    if base_config_path is None:
        base_config_path = base_config_path = '/'.join(config_path.split('/')[:-1])
        base_config_path = f"{base_config_path}/default.yaml"
    
    if base_config_path and os.path.exists(base_config_path):
        base_cfg = load_config(base_config_path)
        cfg = load_config(config_path)
        
        # Recursively update base config with specific config
        def update_dict(base_dict, new_dict):
            for key, value in new_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    update_dict(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        update_dict(base_cfg, cfg)
        return base_cfg
    else:
        return load_config(config_path)