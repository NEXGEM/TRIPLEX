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

from utils import normalize_adata


class STDataset(torch.utils.data.Dataset):
    def __init__(self, 
                mode: str,
                phase: str,
                fold: int,
                data_dir: str,
                gene_type: str = 'mean',
                num_genes: int = 1000,
                num_outputs: int = 300,
                normalize: bool = False,
                cpm=True,
                smooth=True
                ):
        super(STDataset, self).__init__()
        
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
    
        self.mode = mode
        self.phase = phase
        self.norm_param = {'normalize': normalize, 'cpm': cpm, 'smooth': smooth}
        
        data_path = f"{data_dir}/splits/{phase}_{fold}.csv"
        self.ids = self._get_ids(data_path)
        
        self.int2id = dict(enumerate(self.ids))
        
        if not os.path.isfile(f"{data_dir}/{gene_type}_{num_genes}genes.json"):
            raise ValueError(f"{gene_type}_{num_genes}genes.json is not found in {data_dir}")
        
        with open(f"{data_dir}/{gene_type}_{num_genes}genes.json", 'r') as f:
            self.genes = json.load(f)['genes']
        if gene_type == 'mean':
            self.genes = self.genes[:num_outputs]
        
        if phase == 'train':
            self.adata_dict = {_id: self.load_st(_id, **self.norm_param)[:,self.genes] \
                for _id in self.ids}
            
            self.lengths = [len(adata) for adata in self.adata_dict.values()]
            self.cumlen = np.cumsum(self.lengths)
            
            self.transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomApply([transforms.RandomRotation((90, 90))]),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        
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
            img = self.transforms(img)
            
            adata = self.adata_dict[name]
            expression = adata[idx].X
            expression = expression.toarray().squeeze(0) \
                if sparse.issparse(expression) else expression.squeeze(0)
            
            data['img'] = img
            data['label'] = torch.FloatTensor(expression) 
            
        elif self.phase == 'test':
            name = self.int2id[index]
            img = self.load_img(name)
            img = torch.stack([self.transforms(im) for im in img], dim=0)
            
            if os.path.isfile(f"{self.st_dir}/{name}.h5ad"):
                adata = self.load_st(name, **self.norm_param)[:,self.genes]
                
                if self.mode != 'inference':
                    expression = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
                    data['label'] = torch.FloatTensor(expression)
            
            data['img'] = img
            
        return data
        
    def __len__(self):
        if self.phase == 'train':
            return self.cumlen[-1]
        else:
            return len(self.int2id)
        
    def _get_ids(self, data_path):
        if os.path.isfile(data_path):
            data = pd.read_csv(data_path)
            ids = data['sample_id'].to_list()
        else:
            ids = [f for f in os.listdir(f"{self.img_dir}") if f.endswith('.h5')]
            ids = [os.path.splitext(_id)[0] for _id in ids]
        return ids
    
    def load_img(self, name: str, idx: int = None):
        """Load whole slide image of a sample.

        Args:
            name (str): name of a sample
            idx (int): index of a patch.

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
    
    def load_st(self, name: str, normalize: bool = True, cpm=False, smooth=False):
        """Load gene expression data of a sample.

        Args:
            name (str): name of a sample
            normalize (bool): whether to normalize gene expression data.
            cpm (bool): whether to conduct CPM while normalizing gene expression data.
            smooth (bool): whether to smooth gene expression data.
            
        Returns:
            annData: return adata of st data. 
        """
        path = f"{self.st_dir}/{name}.h5ad"
        adata = sc.read_h5ad(path)
        
        if normalize:
            adata = normalize_adata(adata, cpm=cpm, smooth=smooth)
    
        return adata
        

class EGNDataset(STDataset):
    def __init__(self, 
                mode: str,
                phase: str,
                fold: int,
                data_dir: str,
                gene_type: str = 'mean',
                num_genes: int = 1000,
                num_outputs: int = 300,
                normalize: bool = False,
                cpm=True,
                smooth=True
                ):
        super(EGNDataset, self).__init__(
                                mode=mode,
                                phase=phase,
                                fold=fold,
                                data_dir=data_dir,
                                gene_type=gene_type,
                                num_genes=num_genes,
                                num_outputs=num_outputs,
                                normalize=normalize,
                                cpm=cpm,
                                smooth=smooth )
        
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
            img = self.transforms(img)
            
            adata = self.adata_dict[name]
            expression = adata[idx].X
            expression = expression.toarray().squeeze(0) \
                if sparse.issparse(expression) else expression.squeeze(0)
            
                
            data['img'] = img
            data['label'] = expression
            
        elif self.phase == 'test':
            name = self.int2id[index]
            img = self.load_img(name)
            img = torch.stack([self.transforms(im) for im in img], dim=0)
            
            if os.path.isfile(f"{self.st_dir}/{name}.h5ad"):
                adata = self.load_st(name)[:,self.genes]
                
                if self.mode != 'inference':
                    expression = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
                    data['label'] = expression
            
            data['img'] = img
            
        return data
        
    def __len__(self):
        if self.phase == 'train':
            return self.cumlen[-1]
        else:
            return len(self.int2id)
        
        
class BleepDataset(STDataset):
    def __init__(self, 
                mode: str,
                phase: str,
                fold: int,
                data_dir: str,
                gene_type: str = 'mean',
                num_genes: int = 1000,
                num_outputs: int = 300,
                normalize: bool = False,
                cpm=True,
                smooth=True
                ):
        super(BleepDataset, self).__init__(
                                mode=mode,
                                phase=phase,
                                fold=fold,
                                data_dir=data_dir,
                                gene_type=gene_type,
                                num_genes=num_genes,
                                num_outputs=num_outputs,
                                normalize=normalize,
                                cpm=cpm,
                                smooth=smooth )
            
        if mode != 'cv':
            data_path = f"{data_dir}/splits/train_{fold}.csv"
            ids_ref = self._get_ids(data_path)
            
            spot_expressions_ref = []
            for _id in ids_ref:
                expression = self.load_st(_id, **self.norm_param)[:,self.genes].X
                expression = expression.toarray() if sparse.issparse(expression) else expression
                expression = torch.FloatTensor(expression) 
                spot_expressions_ref.append(expression)
                
            self.spot_expressions_ref = torch.cat(spot_expressions_ref, dim=0) 
        
    def __len__(self):
        if self.phase == 'train':
            return self.cumlen[-1]
        else:
            return len(self.int2id)