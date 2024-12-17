
from glob import glob
import os
import json

import numpy as np
from scipy import sparse
import pandas as pd
import h5py
import scanpy as sc
import torch
import torchvision.transforms as transforms

from dataset.base_dataset import BaseDataset


class TriDataset(BaseDataset):
    def __init__(self, 
                mode: str,
                phase: str,
                fold: int,
                data_dir: str,
                gene_type: str = 'mean',
                num_genes: int = 1000,
                num_outputs: int = 300
                ):
        super(TriDataset, self).__init__()
        
        if mode not in ['cv', 'eval', 'inference']:
            raise ValueError(f"mode must be 'cv' or 'eval' or 'inference', but got {mode}")
        
        if phase not in ['train', 'test']:
            raise ValueError(f"phase must be 'train' or 'test', but got {phase}")

        if mode in ['eval', 'inference'] and phase == 'train':
            print(f"mode is {mode} but phase is 'train', so phase is changed to 'test'")
            phase = 'test'
            
        if gene_type not in ['var', 'mean']:
            raise ValueError(f"gene_type must be 'var' or 'mean', but got {gene_type}")
        
        self.data_dir = data_dir
        self.img_dir = f"{data_dir}/patches"
        self.st_dir = f"{data_dir}/adata"
        self.emb_dir = f"{data_dir}/emb"
    
        self.mode = mode
        self.phase = phase
        
        data_path = f"{data_dir}/splits/{phase}_{fold}.csv"
        if os.path.isfile(data_path):
            data = pd.read_csv(data_path)
            ids = data['sample_id'].to_list()
        else:
            ids = [f for f in os.listdir(f"{self.img_dir}") if f.endswith('.h5')]
            ids = [os.path.splitext(_id)[0] for _id in ids]
        
        self.int2id = dict(enumerate(ids))
        
        if not os.path.isfile(f"{data_dir}/{gene_type}_{num_genes}genes.json"):
            raise ValueError(f"{gene_type}_{num_genes}genes.json is not found in {data_dir}")
        
        with open(f"{data_dir}/{gene_type}_{num_genes}genes.json", 'r') as f:
            self.genes = json.load(f)['genes']
        if gene_type == 'mean':
            self.genes = self.genes[:num_outputs]
        
        if phase == 'train':
            self.adata_dict = {_id: self.load_st(_id)[:,self.genes] \
                for _id in ids}
            self.pos_dict = {_id: torch.LongTensor(adata.obs[['array_row', 'array_col']].to_numpy()) \
                for _id, adata in self.adata_dict.items()}
            self.global_embs = {_id: self.load_emb(_id, emb_name='global') \
                for _id in ids}
            
            self.lengths = [len(adata) for adata in self.adata_dict.values()]
            self.cumlen = np.cumsum(self.lengths)
        
    def __getitem__(self, index):
        data = {}
        
        if self.phase == 'train':
            i = 0
            while index >= self.cumlen[i]:
                i += 1
            idx = index
            if i > 0:
                idx = index - self.cumlen[i-1]

            name = self.int2id[i]
            img = self.load_img(name, idx)
            img = self.train_transforms(img)
            
            neighbor_emb, mask = self.load_emb(name, emb_name='neighbor', idx=idx)
            adata = self.adata_dict[name]
            expression = adata[idx].X
            expression = expression.toarray().squeeze(0) \
                if sparse.issparse(expression) else expression.squeeze(0)
            
                
            data['img'] = img
            data['mask'] = mask
            data['neighbor_emb'] = neighbor_emb
            data['label'] = expression
            data['pid'] = torch.LongTensor([i])
            data['sid'] = torch.LongTensor([idx])
            
        elif self.phase == 'test':
            name = self.int2id[index]
            img = self.load_img(name)
            img = torch.stack([self.test_transforms(im) for im in img], dim=0)
            
            global_emb = self.load_emb(name, emb_name='global')
            neighbor_emb, mask = self.load_emb(name, emb_name='neighbor')
            
            if os.path.isfile(f"{self.st_dir}/{name}.h5ad"):
                adata = self.load_st(name)[:,self.genes]
                pos = adata.obs[['array_row', 'array_col']].to_numpy()
                
                if self.mode != 'inference':
                    expression = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
                    data['label'] = expression
            
            else:
                pos = np.load(f"{self.data_dir}/pos/{name}.npy")
            
            data['img'] = img
            data['mask'] = mask
            data['neighbor_emb'] = neighbor_emb
            data['position'] = torch.LongTensor(pos)
            data['global_emb'] = global_emb
            
        return data
        
    def __len__(self):
        if self.phase == 'train':
            return self.cumlen[-1]
        else:
            return len(self.int2id)
        
    def load_emb(self, name: str, emb_name: str = 'global', idx: int = None):
        if emb_name not in ['global', 'neighbor']:
            raise ValueError(f"emb_name must be 'global' or 'neighbor', but got {emb_name}")
        
        path = f"{self.emb_dir}/{emb_name}/uni_v1/{name}.h5"
        
        with h5py.File(path, 'r') as f:
            if 'embeddings'in f:
                emb = f['embeddings'][idx] if idx is not None else f['embeddings'][:]
            else:
                emb = f['features'][idx] if idx is not None else f['features'][:]
                
            emb = torch.Tensor(emb)
            
            if emb_name == 'neighbor':
                mask = f['mask_tb'][idx] if idx is not None else f['mask_tb'][:]
                mask = torch.LongTensor(mask)
                return emb, mask
            
        return emb
    