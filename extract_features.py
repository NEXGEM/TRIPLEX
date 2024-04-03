
import os 
import argparse
from tqdm import tqdm
import sys
from glob import glob

import torch
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn

from models.TRIPLEX import TRIPLEX, load_model_weights
from datasets.st_data import STDataset
from utils import load_config


def get_sub_features(model, patches, num_n):
    neighbor_features = []
    for m in range(num_n):
        for k in range(num_n):
            patch = patches[:,:,224*m:224*(m+1), 224*k:224*(k+1)]
            features = model(patch.to(torch.device('cuda:0')))
            features = features.detach().cpu()
            
            neighbor_features.append(features)
    neighbor_features = torch.stack(neighbor_features).transpose(1,0)
    
    return neighbor_features


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='her2st/TRIPLEX', help='logger path.')
    parser.add_argument('--test_name', type=str, default='10x_breast_ff2', help='dataset name:{"10x_breast_ff1","10x_breast_ff2", "10x_breast_ff3"}.')
    parser.add_argument('--extract_mode', type=str, default='neighbor', help='g_target or neighbor or g_neighbor')
    parser.add_argument('--test_mode', type=str, default='external', help='internal or external')
    parser.add_argument('--num_n', type=int, default=5, help='')
    
    args = parser.parse_args()
    cfg = load_config(args.config)
    
    # Set variable required for data loading 
    data_dir=cfg.DATASET.data_dir
    data=cfg.DATASET.type
    num_n=args.num_n
    cfg.DATASET.num_neighbors=num_n
    num_k=cfg.TRAINING.num_k 
    
    extract_mode = args.extract_mode
    test_mode = args.test_mode
    test_name = args.test_name
    
    if extract_mode not in ["g_target", "neighbor", "g_neighbor"]:
        raise Exception("Invalid extract_mode")
    
    if test_mode not in ["internal", "external"]:
        raise Exception("Invalid test_mode")
    
    # Load pretrained resnet model
    model = load_model_weights("weights/tenpercent_resnet18.ckpt")
    model = model.to(torch.device('cuda:0'))
        
    model.eval()
    
    # Name of directory to store the extracted features
    if extract_mode == 'g_target':
        save_dir = "gt_features"
    elif extract_mode == 'neighbor':
        save_dir = f"n_features_{num_n}"
    elif extract_mode == 'g_neighbor':
        save_dir = f"gn_features_{num_n}"
    
    if test_mode == 'internal':
        # Extract features for cross-validation
        dir_name=f"{data_dir}/{data}/{save_dir}_224"
        os.makedirs(dir_name, exist_ok=True)
        
        for fold in tqdm(range(num_k)):
    
                testset = STDataset(train=False, fold=fold, external_test=(test_mode == 'external'), extract=extract_mode, **cfg.DATASET)
                dataloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)
                
                for i, (patches, _) in enumerate(dataloader):
                    file_name = f"{testset.names[i]}.pt"
                    
                    if extract_mode == 'g_neighbor':
                        patches = torch.load(f"{data_dir}/{data}/n_features_{num_n}/{file_name}").to(torch.device('cuda:0'))
                    elif extract_mode == 'g_target':
                        patches = patches.to(torch.device('cuda:0')).squeeze()
                    else:
                        patches = patches.squeeze()
                    
                    # Process and save features
                    with torch.no_grad():
                        if extract_mode == 'neighbor':
                            features = get_sub_features(model, patches, num_n)
                        else:
                            features = model(patches)
                            features = features.detach().cpu()
    
                        print(f"Saving {file_name}...")
                        torch.save(features, os.path.join(dir_name, file_name))    
                        
    elif test_mode == 'external':   
        
        # Extract features for external test
        dir_name=f"{data_dir}/test/{test_name}/{save_dir}_224"
        os.makedirs(dir_name, exist_ok=True)
        
        testset = STDataset(train=False, external_test=(test_mode == 'external'), extract=extract_mode, test_data = test_name, **cfg.DATASET)
        dataloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)
        
        for i, patches in tqdm(enumerate(dataloader)):
            file_name = f"{testset.names[i]}.pt"
                
            patches = patches.squeeze().split(512,dim=0)
            
            extracted_features = []
            for patch in patches:
                if extract_mode == 'g_neighbor':
                    patch = torch.load(f"{data_dir}/{data}/n_features_{num_n}/{file_name}").to(torch.device('cuda:0'))
                elif extract_mode == 'g_target':
                    patch = patch.to(torch.device('cuda:0')).squeeze()
                else:
                    patch = patch.squeeze()
                
                # Process and save features
                with torch.no_grad():
                    if extract_mode == 'neighbor':
                        features = get_sub_features(model, patch, num_n)
                    else:
                        features = model(patch)
                        features = features.detach().cpu()
                        
                extracted_features.append(features)
            
            extracted_features = torch.cat(extracted_features, dim=0)
            print(f"Saving {file_name}...")
            torch.save(extracted_features, os.path.join(dir_name, file_name))    