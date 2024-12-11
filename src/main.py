
import os
import random
import argparse
from datetime import datetime

import numpy as np
import torch

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy

from models.model_interface import ModelInterface, CustomWriter
from datasets import DataInterface, TriDataset
from utils import ( collate_fn, 
                    load_callbacks, 
                    load_config, 
                    load_loggers, 
                    fix_seed )


def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, default='gbm/TRIPLEX', help='logger path.')
    parser.add_argument('--gpu', type=int, default=[0], help='gpu id')
    parser.add_argument('--mode', type=str, default='cv', help='cv / eval / inference')
    parser.add_argument('--test_name', type=str, default='DRP1', help='dataset name:{"10x_breast_ff1","10x_breast_ff2", "10x_breast_ff3"}.')
    parser.add_argument('--exp_id', type=int, default=0, help='')
    parser.add_argument('--fold', type=int, default=0, help='')
    parser.add_argument('--model_path', type=str, default='logs/2024-04-10/0-TRIPLEX-her2st-2021-3/0-TRIPLEX-her2st-2021-3-epoch=20-valid_loss=0.3268-R=0.1917.ckpt', help='')

    args = parser.parse_args()
    
    return args

def load_dataset(cfg):
    data_loaders = {}
    
    mode = cfg.GENERAL.mode
    batch_size = cfg.TRAINING.batch_size
    data_dir = cfg.DATASET.data_dir
    
    # Load dataset
    if mode == 'cv':
        trainset = TriDataset(mode='cv', 
                            phase='train', 
                            fold=fold, 
                            data_dir=data_dir)
        train_loader = DataLoader(trainset, 
                                batch_size=batch_size, 
                                # collate_fn=collate_fn, 
                                num_workers=4, 
                                pin_memory=True, 
                                shuffle=True)
        data_loaders['train_loader'] = train_loader
        
    testset = TriDataset(mode=mode, 
                        phase='test', 
                        fold=fold, 
                        data_dir=data_dir)
    test_loader = DataLoader(testset, 
                            batch_size=1, 
                            num_workers=4, 
                            pin_memory=True, 
                            shuffle=False)
        
    data_loaders['test_loader'] = test_loader
    
    return data_loaders 


def main(cfg):    
    
    fix_seed(cfg.GENERAL.seed)
    
    # Configurations
    mode = cfg.GENERAL.mode
    gpus = cfg.GENERAL.gpu
    
    # Get dataloader
    data_loaders = load_dataset(cfg)
    
    # Load loggers and callbacks for Trainer
    loggers = load_loggers(cfg)
    callbacks = load_callbacks(cfg)
    
    # Define Data 
    DataInterface_dict = {'dataset_name': cfg.DATA.dataset_name,
                        'dataset_cfg': cfg.DATA}
    dm = DataInterface(**DataInterface_dict)
    
    # Define model
    ModelInterface_dict = {'model': cfg.MODEL,
                            'config': cfg}
    model = ModelInterface(**ModelInterface_dict)
    
    # Train or test model
    if mode == 'cv':
        # Instancialize Trainer 
        trainer = pl.Trainer(
            accelerator="gpu", 
            strategy = DDPStrategy(find_unused_parameters=False),
            devices = gpus,
            max_epochs = cfg.TRAINING.num_epochs,
            logger = loggers,
            check_val_every_n_epoch = 1,
            log_every_n_steps=10,
            callbacks = callbacks,
            amp_backend = 'native',
            precision = 16
        )
        
        trainer.fit(model, datamodule = dm)
        
    elif mode == 'eval':
        trainer = pl.Trainer(accelerator="gpu", devices=gpus)
        
        # checkpoint = glob(f'logs/{log_name}/*.ckpt')[0]
        checkpoint = cfg.GENERAL.model_path
        model = model.load_from_checkpoint(checkpoint, **ModelInterface_dict)
        
        trainer.test(model, datamodule = dm)
        
    elif mode == 'inference':
        pred_path = f"{cfg.DATASET.data_dir}/test/{cfg.GENERAL.test_name}/pred_{cfg.fold}"
        emb_path = f"{cfg.DATASET.data_dir}/test/{cfg.GENERAL.test_name}/emb_{cfg.fold}"
        
        os.makedirs(pred_path, exist_ok=True)
        os.makedirs(emb_path, exist_ok=True)
        
        names = data_loaders['test_loader'].dataset.names
        pred_writer = CustomWriter(pred_dir=pred_path, emb_dir=emb_path, write_interval="epoch", names=names)
        trainer = pl.Trainer(accelerator="gpu", devices=gpus, callbacks=[pred_writer])

        checkpoint = cfg.GENERAL.model_path
        model = model.load_from_checkpoint(checkpoint, **ModelInterface_dict)
        
        trainer.predict(model, datamodule = dm, return_predictions=False)
        
    else:
        raise Exception("Invalid mode")
    
    return model

if __name__ == '__main__':
    args = get_parse()   
    cfg = load_config(args.config_name)

    cfg.config = args.config_name
    cfg.GENERAL.test_name = args.test_name
    cfg.GENERAL.exp_id = args.exp_id
    cfg.GENERAL.gpu = args.gpu
    cfg.GENERAL.model_path = args.model_path
    cfg.DATA.mode = args.mode

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.GENERAL.timestamp = timestamp
    
    if args.mode == 'cv':
        num_k = cfg.TRAINING.num_k     
        for fold in range(num_k):
            cfg.DATA.fold = fold
            main(cfg)
    else:
        cfg.DATA.fold = args.fold
        main(cfg)
