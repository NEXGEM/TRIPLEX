

import random
import argparse
from glob import glob
import time

import numpy as np
import pandas as pd 
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy

from models.TRIPLEX import TRIPLEX
from datasets.st_data import STDataset
from utils import collate_fn, load_callbacks, load_config, load_loggers


parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, default='her2st/TRIPLEX', help='logger path.')
parser.add_argument('--gpu', type=int, default=1, help='number of gpus')
parser.add_argument('--mode', type=str, default='cv', help='cv / test / ex_test')
parser.add_argument('--test_name', type=str, default='10x_breast_ff1', help='dataset name:{"10x_breast_ff1","10x_breast_ff2", "10x_breast_ff3"}.')
parser.add_argument('--exp_id', type=int, default=0, help='')
parser.add_argument('--fold', type=int, default=0, help='')
parser.add_argument('--num_n', type=int, default=5, help='')

args = parser.parse_args()
cfg = load_config(args.config_name)

cfg.DATASET.num_neighbors = args.num_n

name=cfg.MODEL.name
seed=cfg.GENERAL.seed
data=cfg.DATASET.type
batch_size=cfg.TRAINING.batch_size
num_k=cfg.TRAINING.num_k 
num_epochs=cfg.TRAINING.num_epochs


random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def main(mode='cv', fold=0, name=None):    
    # Load dataset
    if mode == 'cv' or mode == 'cal_train_time':
        trainset = STDataset(train=True, fold=fold, **cfg.DATASET)
        train_loader = DataLoader(trainset, batch_size=batch_size, collate_fn=collate_fn, num_workers=8, pin_memory=True, shuffle=True)
    
    if mode == 'ex_test':
        testset = STDataset(train=False, fold=fold, external_test=(mode=='ex_test'), test_data=args.test_name, **cfg.DATASET)
        test_loader = DataLoader(testset, batch_size=1, num_workers=8, pin_memory=True, shuffle=False)
        
    else:
        testset = STDataset(train=False, fold=fold, external_test=(mode=='ex_test'), **cfg.DATASET)
        test_loader = DataLoader(testset, batch_size=1, num_workers=8, pin_memory=True, shuffle=False)
            

    log_name=f'{fold}-{name}-{data}-{seed}-{args.exp_id}'
    # Set log name
    log_name=f'{fold}-{name}-{data}-{seed}-{args.exp_id}'
    
    cfg.GENERAL.log_name = log_name
    if mode == 'cv':
        print(log_name)

    # Load loggers and callbacks for Trainer
    loggers = load_loggers(cfg)
    callbacks = load_callbacks(cfg)
    
    model_cfg = cfg.MODEL.copy()
    del model_cfg['name']
    
    # Load model
    model = TRIPLEX(res_neighbor=(args.num_n,args.num_n),**model_cfg)
    
    # Instancialize Trainer 
    trainer = pl.Trainer(
        accelerator="gpu", 
        strategy = DDPStrategy(find_unused_parameters=False),
        devices = args.gpu,
        max_epochs = num_epochs,
        logger = loggers,
        check_val_every_n_epoch = 1,
        log_every_n_steps=10,
        callbacks = callbacks,
        amp_backend = 'native',
        precision = 16
    )
        
    # Train or test model
    if mode == 'cv' or mode == 'cv_each':
        trainer.fit(model, train_loader, test_loader)
        
    elif mode == 'ex_test':
        checkpoint = glob(f'logs/{log_name}/*.ckpt')[0]
        model = model.load_from_checkpoint(checkpoint, ab_id=args.ab_id, res_neighbor=(args.num_n,args.num_n), **model_cfg)
        
        trainer.test(model, test_loader)
        
    elif mode=='test':
        checkpoint = glob(f'logs/{log_name}/*.ckpt')[0]
        model = model.load_from_checkpoint(checkpoint, ab_id=args.ab_id, res_neighbor=(args.num_n,args.num_n), **model_cfg)
        
        trainer.test(model, test_loader)
        
    else:
        raise Exception("Invalid mode")

if __name__ == '__main__':
        
    if args.mode in ['cv', 'test']:
        
        for fold in range(num_k):
            main(fold=fold, mode=args.mode, name=name)
    
    elif args.mode == "cv_each":
        main(fold=args.fold, mode=args.mode, name=name)
    
    else:
        main(fold=args.fold, mode=args.mode, name=name)
