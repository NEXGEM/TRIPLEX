

import os

from datetime import datetime
import yaml
from addict import Dict
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd

import torch
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback



def match(x: np.array, y: np.array):
    """Returns a NumPy array of the positions of first occurrences of values in y in x. 

    Returns:
        np.array: a NumPy array of the positions of first occurrences of values in y in x
    """
    positions = np.empty_like(y, dtype=np.float64)
    for i, value in np.ndenumerate(y):
        try:
            positions[i] = np.where(x == value)[0][0]
        except IndexError:
            positions[i] = np.nan
    return positions.astype('int16')


def smooth_exp(cnt: pd.DataFrame):
    """Apply smoothing to gene expression data in Pandas DataFrame.
    Take average gene expression of the nearest 9 spots.
    
    Args:
        cnt (pd.DataFrame): count data 

    Returns:
        pd.DataFrame: smoothed expression in DataFrame. 
    """

    ids = cnt.index
    delta = np.array([[1,0],
            [0,1],
            [-1,0],
            [0,-1],
            [1,1],
            [-1,-1],
            [1,-1],
            [-1,1],
            [0,0]])

    cnt_smooth = np.zeros_like(cnt).astype('float')

    for i in range(len(cnt)):
        spot = cnt.iloc[i,:]    
        
        # print(f"Smoothing {spot.name}")    
        center = np.array(spot.name.split('x')).astype('int')
        neighbors = center - delta
        neighbors = pd.DataFrame(neighbors).astype('str').apply(lambda x: "x".join(x), 1)
        
        cnt_smooth[i,:] = cnt[ids.isin(neighbors)].mean(0)
        
    cnt_smooth = pd.DataFrame(cnt_smooth)
    cnt_smooth.columns = cnt.columns
    cnt_smooth.index = cnt.index
    
    return cnt_smooth

def collate_fn(batch: tuple):
    """Custom collate function of train dataloader for TRIPLEX.   

    Args:
        batch (tuple): batch of returns from Dataset

    Returns:
        tuple: batch data
    """
    
    patch = torch.stack([item[0] for item in batch])
    exp = torch.stack([item[1] for item in batch])
    pid = torch.stack([item[2] for item in batch])
    sid = torch.stack([item[3] for item in batch])
    wsi = [item[4] for item in batch]
    position = [item[5] for item in batch]
    neighbors = torch.stack([item[6] for item in batch])
    mask = torch.stack([item[7] for item in batch])
    
    return patch, exp, pid, sid, wsi, position, neighbors, mask

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
    
    log_path = os.path.join(cfg.GENERAL.log_path, cfg.GENERAL.current_day)

    tb_logger = TensorBoardLogger(
        log_path,
        name = cfg.GENERAL.log_name)

    csv_logger = CSVLogger(
        log_path,
        name = cfg.GENERAL.log_name)
    
    loggers = [tb_logger, csv_logger]
    
    return loggers

# load Callback
def load_callbacks(cfg: Dict):
    """Return Early stopping and Checkpoint Callbacks. 

    Args:
        cfg (Dict): Dict containing configuration.

    Returns:
        List: Return List containing the Callbacks.
    """
    log_path = os.path.join(cfg.GENERAL.log_path, cfg.GENERAL.current_day)
    
    Mycallbacks = []
    
    target = cfg.TRAINING.early_stopping.monitor
    patience = cfg.TRAINING.early_stopping.patience
    mode = cfg.TRAINING.early_stopping.mode
    
    
    early_stop_callback = EarlyStopping(
        monitor=target,
        min_delta=0.00,
        patience=patience,
        verbose=True,
        mode=mode
    )
    Mycallbacks.append(early_stop_callback)
    fname = cfg.GENERAL.log_name + '-{epoch:02d}-{valid_loss:.4f}' if cfg.MODEL.name == "BLEEP" else cfg.GENERAL.log_name + '-{epoch:02d}-{valid_loss:.4f}-{R:.4f}'
    checkpoint_callback = ModelCheckpoint(monitor = target,
                                    dirpath = str(log_path) + '/' + cfg.GENERAL.log_name,
                                    filename=fname,
                                    verbose = True,
                                    save_last = False,
                                    save_top_k = 1,
                                    mode = mode,
                                    save_weights_only = True)
    Mycallbacks.append(checkpoint_callback)
        
    return Mycallbacks
