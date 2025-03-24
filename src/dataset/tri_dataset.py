
import os

import numpy as np
from scipy import sparse
import h5py
import torch

from dataset.base_dataset import STDataset


class TriDataset(STDataset):
    def __init__(self, 
                mode: str,
                phase: str,
                fold: int,
                data_dir: str,
                gene_type: str = 'mean',
                num_genes: int = 1000,
                num_outputs: int = 300,
                normalize: bool = True,
                cpm: bool = False,
                smooth: bool = False,
                data_id: str = None,
                model_name: str = 'uni_v1'
                ):
        super(TriDataset, self).__init__(
                                mode=mode,
                                phase=phase,
                                fold=fold,
                                data_dir=data_dir,
                                gene_type=gene_type,
                                num_genes=num_genes,
                                num_outputs=num_outputs,
                                normalize=normalize,
                                cpm=cpm,
                                smooth=smooth,
                                data_id=data_id)
    
        self.emb_dir = f"{data_dir}/emb"
        
        if phase == 'train':
            self.pos_dict = {_id: torch.LongTensor(adata.obs[['array_row', 'array_col']].to_numpy()) \
                for _id, adata in self.adata_dict.items()}
            self.global_embs = {_id: self.load_emb(_id, emb_name='global', model_name=model_name) \
                for _id in self.ids}
        
        if mode == 'inference':
            self.global_emb = self.load_emb(self.name, emb_name='global')
            self.position = torch.LongTensor(np.load(f"{self.data_dir}/pos/{self.name}.npy")) 
        
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
            
            neighbor_emb, mask = self.load_emb(name, emb_name='neighbor', idx=idx)
            adata = self.adata_dict[name]
            expression = adata[idx].X
            expression = expression.toarray().squeeze(0) \
                if sparse.issparse(expression) else expression.squeeze(0)
            
                
            data['img'] = img
            data['mask'] = mask
            data['neighbor_emb'] = neighbor_emb
            data['label'] = torch.FloatTensor(expression)
            data['pid'] = torch.LongTensor([i])
            data['sid'] = torch.LongTensor([idx])
            
        elif self.phase == 'test':
            if self.mode == 'inference':
                img = self.img[index]
                img = self.transforms(img)
                neighbor_emb, mask = self.load_emb(self.name, emb_name='neighbor', idx=index)
                # global_emb = self.load_emb(self.name, emb_name='global')
                # pos = np.load(f"{self.data_dir}/pos/{self.name}.npy")
                data['sid'] = torch.LongTensor([index])
            else:
                name = self.int2id[index]
                img = self.load_img(name)
                img = torch.stack([self.transforms(im) for im in img], dim=0)
                
                neighbor_emb, mask = self.load_emb(name, emb_name='neighbor')
                global_emb = self.load_emb(name, emb_name='global')
            
                if os.path.isfile(f"{self.st_dir}/{name}.h5ad"):
                    adata = self.load_st(name, self.genes, **self.norm_param)
                    pos = adata.obs[['array_row', 'array_col']].to_numpy()
                    
                    expression = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
                    data['label'] = torch.FloatTensor(expression) 
                    # if self.mode != 'inference':
                    #     expression = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
                    #     data['label'] = torch.FloatTensor(expression) 
                else:
                    pos = np.load(f"{self.data_dir}/pos/{name}.npy")
                    
                data['position'] = torch.LongTensor(pos)
                data['global_emb'] = global_emb
                    
            
            data['img'] = img
            data['mask'] = mask
            data['neighbor_emb'] = neighbor_emb
            
        return data
        
    def load_emb(self, name: str, emb_name: str = 'global', idx: int = None, model_name: str = 'uni_v1'):
        if emb_name not in ['global', 'neighbor']:
            raise ValueError(f"emb_name must be 'global' or 'neighbor', but got {emb_name}")
        
        path = f"{self.emb_dir}/{emb_name}/{model_name}/{name}.h5"
        
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
    
