
import os
import argparse
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy

from models import ModelInterface, CustomWriter
from datasets import DataInterface
from utils import ( load_callbacks, 
                    load_config, 
                    load_loggers, 
                    fix_seed )

torch.set_float32_matmul_precision('high')

def get_parse():
    parser = argparse.ArgumentParser()
    
    # Main configuration
    parser.add_argument('--config_name', type=str, default='gbm/TRIPLEX', help='logger path.')
    parser.add_argument('--mode', type=str, default='inference', help='cv / eval / inference')
    # Acceleration 
    parser.add_argument('--gpu', type=int, default=1, help='gpu id')
    # Experiments
    parser.add_argument('--exp_id', type=int, default=0, help='')
    # Others
    parser.add_argument('--fold', type=int, default=2, help='')
    parser.add_argument('--ckpt_path', type=str, default='weights/TRIPLEX/epoch=16-val_MeanSquaredError=0.4553.ckpt', help='')

    args = parser.parse_args()
    
    return args

def main(cfg):    
    
    fix_seed(cfg.GENERAL.seed)
    
    # Configurations
    mode = cfg.DATA.mode
    gpus = cfg.GENERAL.gpu
    
    # Load loggers and callbacks for Trainer
    loggers = load_loggers(cfg)
    callbacks = load_callbacks(cfg)
    
    # Define Data 
    DataInterface_dict = {'dataset_name': cfg.DATA.dataset_name,
                        'data_config': cfg.DATA}
    dm = DataInterface(**DataInterface_dict)
    
    # Define model
    ModelInterface_dict = {'model_name': cfg.MODEL.model_name,
                            'config': cfg}
    
    # Train or test model
    if mode == 'cv':

        model = ModelInterface(**ModelInterface_dict)
        
        # Instancialize Trainer 
        trainer = pl.Trainer(
            accelerator="gpu", 
            strategy = DDPStrategy(find_unused_parameters=False),
            devices = gpus,
            max_epochs = cfg.TRAINING.num_epochs,
            logger = loggers,
            check_val_every_n_epoch = 1,
            callbacks = callbacks,
            precision = '16-mixed'
        )
        
        trainer.fit(model, datamodule = dm)
        
    elif mode == 'eval':
        trainer = pl.Trainer(accelerator="gpu", 
                            devices=gpus,
                            precision = '16-mixed',
                            logger=False)
        
        checkpoint = cfg.GENERAL.ckpt_path
        model = ModelInterface.load_from_checkpoint(checkpoint, **ModelInterface_dict)
        
        trainer.test(model, datamodule = dm)
        
    elif mode == 'inference':
        pred_path = f"{cfg.DATA.output_dir}/pred/fold{cfg.DATA.fold}"
        os.makedirs(pred_path, exist_ok=True)
        
        pred_writer = CustomWriter(pred_dir=pred_path, write_interval="epoch")
        trainer = pl.Trainer(accelerator="gpu", 
                            devices=gpus, 
                            callbacks=[pred_writer],
                            precision = '16-mixed',
                            logger=False)

        model = ModelInterface.load_from_checkpoint(cfg.MODEL.ckpt_path, **ModelInterface_dict)
        
        trainer.predict(model, datamodule = dm, return_predictions=False)
        
    else:
        raise Exception("Invalid mode")
    
    return model

if __name__ == '__main__':
    args = get_parse()   
    cfg = load_config(args.config_name)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    cfg.GENERAL.timestamp = timestamp
    cfg.config = args.config_name
    cfg.GENERAL.exp_id = args.exp_id
    cfg.GENERAL.gpu = args.gpu
    cfg.MODEL.ckpt_path = args.ckpt_path
    cfg.DATA.mode = args.mode
    
    if args.mode == 'cv':
        num_k = cfg.TRAINING.num_k     
        for fold in range(num_k):
            cfg.DATA.fold = fold
            main(cfg)
    else:
        cfg.DATA.fold = args.fold
        main(cfg)
