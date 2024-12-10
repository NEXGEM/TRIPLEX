
from glob import glob
import os

import numpy as np
import pandas as pd
import h5py
import scanpy as sc
import torch
import torchvision.transforms as transforms


class STDataset(torch.utils.data.Dataset):
    """Some Information about baselines"""
    def __init__(self):
        super(STDataset, self).__init__()
        
        self.train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([transforms.RandomRotation((90, 90))]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        
        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def load_img(self, name: str, idx: int = None):
        """Load whole slide image of a sample.

        Args:
            name (str): name of a sample

        Returns:
            numpy.array: return whole slide image.
        """
        path = f"{self.img_dir}/{name}.h5"
        
        if idx is not None:
            with h5py.File(path, 'r') as f:
                img = f['img'][idx]
        else:
            with h5py.File(path, 'r') as f:
                img = f['img'][:]
            
        return img
    
    def load_st(self, name: str):
        """Load gene expression data of a sample.

        Args:
            name (str): name of a sample

        Returns:
            annData: return adata of st data. 
        """
        path = f"{self.st_dir}/{name}.h5ad"
        adata = sc.read_h5ad(path)
    
        return adata


class TriDataset(STDataset):
    def __init__(self, 
                mode: str,
                phase: str,
                fold: int,
                data_dir: str,
                ):
        super(TriDataset, self).__init__()
        
        assert mode in ['cv', 'inference'], f"mode must be 'cv' or 'inference', but got {mode}"
        assert phase in ['train', 'test'], f"phase must be 'train' or 'test', but got {phase}"

        if mode == 'inference':
            phase = 'test'
        
        self.img_dir = f"{data_dir}/patches"
        self.st_dir = f"{data_dir}/st"
        self.emb_dir = f"{data_dir}/emb"
        # self.global_dir = f"{data_dir}/emb/global"
        # self.neighbor_dir = f"{data_dir}/emb/neighbor"
    
        self.mode = mode
        self.phase = phase
        
        if mode == 'cv':
            data_path = f"{data_dir}/splits/{phase}_fold{fold}.csv"
            data = pd.read_csv(data_path)
            ids = data['sample_id'].to_list()
                
        elif mode == 'inference':
            ids = os.listdir(f"{self.img_dir}")
            ids = [os.path.splitext(_id)[0] for _id in ids]
            
        self.int2id = dict(enumerate(ids))
        
        if phase == 'train':
            self.st_dict = {_id: self.load_st(_id) for _id in ids}
            # self.img_dict = {_id: self.load_img(_id) for _id in ids}
            # self.emb_dict = {_id: self.load_emb(_id) for _id in ids}

            self.lengths = [len(adata) for adata in self.st_dict.values()]
            self.cumlen = np.cumsum(self.lengths)
        
    def __getitem__(self, index):
        if self.phase == 'train':
            i = 0
            while index >= self.cumlen[i]:
                i += 1
            idx = index
            if i > 0:
                idx = index - self.cumlen[i-1]

            name = self.id2name[i]
            img = self.load_img(name, idx)
            global_emb = self.load_emb(name, emb_name='global')
            neighbor_emb = self.load_emb(name, emb_name='neighbor', idx=idx)
            
            adata = self.st_dict[name]
            pos = adata.obs[['array_row', 'array_col']].to_numpy()
            st = adata[idx]
            
        elif self.phase == 'test':
            name = self.id2name[index]
            img = self.load_img(name)
            global_emb = self.load_emb(name, emb_name='global')
            neighbor_emb = self.load_emb(name, emb_name='neighbor')
            
            if self.mode == 'cv':
                st = self.load_st(name)
                pos = adata.obs[['array_row', 'array_col']].to_numpy()
            
            elif self.mode == 'inference':
                pos = np.load(f"{self.data_dir}/pos/{name}.npy")
                
        if self.mode == 'cv':
            return {'img': img, 
                    'st': st, 
                    'pos': pos, 
                    'global_emb': global_emb, 
                    'neighbor_emb': neighbor_emb}
            
        elif self.mode == 'inference':
            return {'img': img, 
                    'pos': pos, 
                    'global_emb': global_emb, 
                    'neighbor_emb': neighbor_emb}
        
        
    def __len__(self):
        if self.mode == 'train':
            return self.cumlen[-1]
        else:
            return len(self.int2id)
        
    def load_emb(self, name: str, emb_name: str = 'global', idx: int = None):
        assert emb_name in ['global', 'neighbor'], f"emb_name must be 'global' or 'neighbor', but got {emb_name}"
    
        path = f"{self.emb_dir}/{emb_name}/{name}.h5"
        
        if idx is not None:
            with h5py.File(path, 'r') as f:
                emb = f['embeddings'][idx]
        else:
            with h5py.File(path, 'r') as f:
                emb = f['embeddings'][:]
                
        return emb
