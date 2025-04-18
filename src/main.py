
import os
import argparse
from datetime import datetime
from glob import glob 
from tqdm import tqdm
from pathlib import Path
from shutil import copyfile

import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning import loggers as pl_loggers

from model import ModelInterface, CustomWriter
from dataset import DataInterface
from utils.train_utils import ( load_callbacks, 
                    load_config, 
                    load_loggers, 
                    fix_seed )

torch.set_float32_matmul_precision('high')

def get_parse():
    parser = argparse.ArgumentParser()
    
    # Main configuration
    parser.add_argument('--config_name', type=str, default='hest/bench_data/LYMPH_IDC/TRIPLEX', help='Path to the configuration file for the experiment.')
    parser.add_argument('--mode', type=str, default='cv', help='Mode of operation: "cv" for cross-validation, "eval" for evaluation, "inference" for inference')
    # Acceleration 
    parser.add_argument('--gpu', type=int, default=1, help='Number of gpus to use')
    # Experiments
    parser.add_argument('--exp_id', type=int, default=0, help='Experiment ID for tracking different runs')
    # Others
    parser.add_argument('--fold', type=int, default=0, help='Fold number for cross-validation')
    parser.add_argument('--ckpt_path', type=str, default='weights/TRIPLEX/epoch=25-val_target=0.5430.ckpt', help='Path to the checkpoint file for model weights')
    parser.add_argument('--timestamp', type=str, default=None, help='timestamp name for the loggers')

    args = parser.parse_args()
    
    return args

def main(cfg):    
    
    fix_seed(cfg.GENERAL.seed)
    
    # Configurations
    mode = cfg.DATA.mode
    gpus = cfg.GENERAL.gpu
    use_amp = cfg.GENERAL.get('use_amp', True)
    
    # Define Data 
    if mode != 'inference':
        DataInterface_dict = {'dataset_name': cfg.DATA.dataset_name,
                            'data_config': cfg.DATA}
        dm = DataInterface(**DataInterface_dict)
        
    # Define model
    ModelInterface_dict = {'model_name': cfg.MODEL.model_name,
                            'config': cfg}
    
    # Train or test model
    if mode == 'cv':
        # Load loggers and callbacks for Trainer
        loggers = load_loggers(cfg)
        callbacks = load_callbacks(cfg)
        
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
            precision = '16-mixed' if use_amp else '32'
        )
        
        trainer.fit(model, datamodule = dm)
        
    elif mode == 'eval':
        log_path = cfg.GENERAL.log_path
    
        ckpt_dir = f'{log_path}/{cfg.config}/{cfg.GENERAL.timestamp}'    
        ckpt_path = glob(f"{ckpt_dir}/fold{cfg.DATA.fold}/*.ckpt")[0]
        
        log_name = str(Path(cfg.config).parent)
        model_name = Path(cfg.config).name
        current_time = Path(ckpt_dir).name
        
        output_path = f"{cfg.DATA.output_dir}/{model_name}"
        os.makedirs(f"{output_path}/fold{cfg.DATA.fold}", exist_ok=True)
        cfg.DATA.output_path = output_path
    
        csv_logger = pl_loggers.CSVLogger(f"{log_path}/{log_name}",
                                    name = f"{model_name}/{current_time}", version = f'fold{cfg.DATA.fold}/eval', )
        
        trainer = pl.Trainer(accelerator="gpu", 
                            devices=gpus,
                            precision = '16-mixed' if use_amp else '32',
                            logger=[csv_logger])
        
        # checkpoint = cfg.GENERAL.ckpt_path
        model = ModelInterface.load_from_checkpoint(ckpt_path, **ModelInterface_dict)
        
        trainer.test(model, datamodule = dm)
        
    elif mode == 'inference':
        model_name = Path(cfg.config).name
        
        pred_path = f"{cfg.DATA.output_dir}/{model_name}/fold{cfg.DATA.fold}"
        os.makedirs(pred_path, exist_ok=True)
        
        pred_writer = CustomWriter(pred_dir=pred_path, write_interval="epoch")
        trainer = pl.Trainer(accelerator="gpu", 
                            devices=gpus, 
                            callbacks=[pred_writer],
                            precision = '16-mixed' if use_amp else '32',
                            logger=False)

        model = ModelInterface.load_from_checkpoint(cfg.MODEL.ckpt_path, **ModelInterface_dict)
        
        ids = os.listdir(f"{cfg.DATA.data_dir}/patches")
        ids = [_id.split('.')[0] for _id in ids if _id.endswith('.h5')]
        for _id in tqdm(ids):
            if os.path.isfile(f"{pred_path}/{_id}.pt"):
                print("Already predicted", _id)
                continue
            
            print("Predicting", _id)
            cfg.DATA.data_id = _id
            DataInterface_dict = {'dataset_name': cfg.DATA.dataset_name,
                            'data_config': cfg.DATA}
            dm = DataInterface(**DataInterface_dict)
        
            trainer.predict(model, datamodule = dm, return_predictions=False)
        
    else:
        raise Exception("Invalid mode")
    
    return model

if __name__ == '__main__':
    args = get_parse()   
    
    config_path = os.path.join('./config', f'{args.config_name}.yaml')
    cfg = load_config(config_path)
    log_path = cfg.GENERAL.log_path
    Path(log_path).mkdir(exist_ok=True, parents=True)
    log_name = str(Path(args.config_name).parent)
    version_name = Path(args.config_name).name
    
    ## Train setup
    if args.mode == 'cv':
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        cfg.GENERAL.timestamp = timestamp    
        cfg.log_dir = f"{log_path}/{log_name}/{version_name}/{timestamp}"
        os.makedirs(cfg.log_dir, exist_ok=True)
        
        # Copy config file to log path
        copyfile(config_path, f"{cfg.log_dir}/config.yaml")
        
    ## eval setup
    else:
        timestamp = args.timestamp
        if timestamp is None:
            timestamp = sorted(os.listdir(f"{log_path}/{log_name}/{version_name}"))[-1]
        
        config_path = f"{log_path}/{log_name}/{version_name}/{timestamp}/config.yaml"
        cfg = load_config(config_path)
        
    cfg.GENERAL.timestamp = timestamp
    cfg.config = args.config_name
    cfg.GENERAL.exp_id = args.exp_id
    cfg.GENERAL.gpu = args.gpu
    cfg.DATA.mode = args.mode
        
    if args.mode != 'inference':
        num_k = cfg.TRAINING.num_k     
        for fold in range(num_k):
            cfg.DATA.fold = fold
            main(cfg)
    else:
        cfg.MODEL.ckpt_path = args.ckpt_path
        cfg.DATA.fold = args.fold
        main(cfg)
