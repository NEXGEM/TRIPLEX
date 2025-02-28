
import os 
import inspect
import importlib

import wget
import numpy as np
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl

import torch.nn.functional as F
from einops import rearrange

from model.TRIPLEX.module import ( GlobalEncoder, 
                                NeighborEncoder, 
                                FusionEncoder )

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def load_model_weights(ckpt: str):       
        """Load pretrained ResNet18 model without final fc layer.

        Args:
            path (str): path_for_pretrained_weight

        Returns:
            torchvision.models.resnet.ResNet: ResNet model with pretrained weight
        """
        
        resnet = torchvision.models.__dict__['resnet18'](weights=None)
        
        ckpt_dir = './weights'
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = f'{ckpt_dir}/{ckpt}'
        
        # prepare the checkpoint
        if not os.path.exists(ckpt_path):
            ckpt_url='https://github.com/ozanciga/self-supervised-histopathology/releases/download/tenpercent/tenpercent_resnet18.ckpt'
            wget.download(ckpt_url, out=ckpt_dir)
            
        state = torch.load(ckpt_path)
        state_dict = state['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
        
        model_dict = resnet.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        if state_dict == {}:
            print('No weight could be loaded..')
        model_dict.update(state_dict)
        resnet.load_state_dict(model_dict)
        resnet.fc = nn.Identity()

        return resnet


class TRIPLEX(nn.Module):
    """Model class for TRIPLEX
    """
    def __init__(self, 
                num_outputs=250,
                emb_dim=512,
                depth1=2,
                depth2=2,
                depth3=2,
                num_heads1=8,
                num_heads2=8,
                num_heads3=8,
                mlp_ratio1=2.0,
                mlp_ratio2=2.0,
                mlp_ratio3=2.0,
                dropout1=0.1,
                dropout2=0.1,
                dropout3=0.1,
                kernel_size=3,
                res_neighbor=(5,5),
                max_batch_size=1024):
        """TRIPLEX model 

        Args:
            num_genes (int): Number of genes to predict.
            emb_dim (int): Embedding dimension for images. Defaults to 512.
            depth1 (int): Depth of FusionEncoder. Defaults to 2.
            depth2 (int): Depth of GlobalEncoder. Defaults to 2.
            depth3 (int): Depth of NeighborEncoder. Defaults to 2.
            num_heads1 (int): Number of heads for FusionEncoder. Defaults to 8.
            num_heads2 (int): Number of heads for GlobalEncoder. Defaults to 8.
            num_heads3 (int): Number of heads for NeighborEncoder. Defaults to 8.
            mlp_ratio1 (float): mlp_ratio (MLP dimension/emb_dim) for FusionEncoder. Defaults to 2.0.
            mlp_ratio2 (float): mlp_ratio (MLP dimension/emb_dim) for GlobalEncoder. Defaults to 2.0.
            mlp_ratio3 (float): mlp_ratio (MLP dimension/emb_dim) for NeighborEncoder. Defaults to 2.0.
            dropout1 (float): Dropout rate for FusionEncoder. Defaults to 0.1.
            dropout2 (float): Dropout rate for GlobalEncoder. Defaults to 0.1.
            dropout3 (float): Dropout rate for NeighborEncoder. Defaults to 0.1.
            kernel_size (int): Kernel size of convolution layer in PEGH. Defaults to 3.
            res_neighbor (tuple): Resolution of neighbor embeddings. Defaults to (5,5).
            max_batch_size (int): Maximum batch size for inference. Defaults to 1024.
        """
        
        super().__init__()
        
        self.alpha = 0.3
        self.emb_dim = emb_dim
        self.max_batch_size = max_batch_size
    
        # Target Encoder
        resnet18 = load_model_weights("tenpercent_resnet18.ckpt")
        module=list(resnet18.children())[:-2]
        self.target_encoder = nn.Sequential(*module)
        self.fc_target = nn.Linear(emb_dim, num_outputs)
        # self.fc_target = nn.Sequential(nn.Linear(emb_dim, num_outputs),
        #                         nn.ReLU())
        self.target_linear = nn.Linear(512, emb_dim)

        # Neighbor Encoder
        self.neighbor_encoder = NeighborEncoder(emb_dim, 
                                                depth3, 
                                                num_heads3, 
                                                int(emb_dim*mlp_ratio3), 
                                                dropout = dropout3, 
                                                resolution=res_neighbor)
        self.fc_neighbor = nn.Linear(emb_dim, num_outputs)
        # self.fc_neighbor = nn.Sequential(nn.Linear(emb_dim, num_outputs),
                                # nn.ReLU())

        # Global Encoder        
        self.global_encoder = GlobalEncoder(emb_dim, 
                                            depth2, 
                                            num_heads2, 
                                            int(emb_dim*mlp_ratio2), 
                                            dropout2, 
                                            kernel_size)
        self.fc_global = nn.Linear(emb_dim, num_outputs)
        # self.fc_global = nn.Sequential(nn.Linear(emb_dim, num_outputs),
                                # nn.ReLU())
    
        # Fusion Layer
        self.fusion_encoder = FusionEncoder(emb_dim, 
                                            depth1, 
                                            num_heads1, 
                                            int(emb_dim*mlp_ratio1), 
                                            dropout1)    
        # self.fc = nn.Sequential(nn.Linear(emb_dim, num_outputs),
        #                         nn.ReLU())
        self.fc = nn.Linear(emb_dim, num_outputs)
        
    def forward(self,
                img, 
                mask, 
                neighbor_emb, 
                position=None, 
                global_emb=None, 
                pid=None, 
                sid=None, 
                **kwargs):
        
        if 'dataset' in kwargs:
            # Training
            return self._process_training_batch(img, mask, neighbor_emb, pid, sid, kwargs['dataset'], kwargs['label'])
        else:
            # Inference 
            return self._process_inference_batch(img, mask, neighbor_emb, position, global_emb, sid)
            
    def _process_training_batch(self, img, mask, neighbor_emb, pid, sid, dataset, label):
        global_emb, position = self.retrieve_global_emb(pid, dataset)
        
        fusion_token, target_token, neighbor_token, global_token = \
            self._encode_all(img, mask, neighbor_emb, position, global_emb, pid, sid)
        return self._get_outputs(fusion_token, target_token, neighbor_token, global_token, label)

    def _process_inference_batch(self, img, mask, neighbor_emb, position, global_emb, sid=None):
        if sid is None and img.shape[0] > self.max_batch_size:
            imgs = img.split(self.max_batch_size, dim=0)
            neighbor_embs = neighbor_emb.split(self.max_batch_size, dim=0)
            masks = mask.split(self.max_batch_size, dim=0)
            sid = torch.arange(img.shape[0]).to(img.device)
            sids = sid.split(self.max_batch_size, dim=0)
            
            pred = [self.fc(self._encode_all(img, mask, neighbor_emb, position, global_emb, sid=sid)[0]) \
                for img, neighbor_emb, mask, sid in zip(imgs, neighbor_embs, masks, sids)]
            return {'logits': torch.cat(pred, dim=0)}    
        else:
            fusion_token, _, _, _ = self._encode_all(img, mask, neighbor_emb, position, global_emb, sid=sid)
            logits = self.fc(fusion_token)
        return {'logits': logits}
    
    def _encode_all(self, img, mask, neighbor_emb, position, global_emb, pid=None, sid=None):
        target_token = self.encode_target(img)
        neighbor_token = self.neighbor_encoder(neighbor_emb, mask)
        global_token = self.encode_global(global_emb, position, pid, sid)
        
        fusion_token = self.fusion_encoder(target_token, neighbor_token, global_token, mask=mask)
        
        return fusion_token, target_token, neighbor_token, global_token
    
    def encode_target(self, img):
        # Target tokens
        target_token = self.target_encoder(img) # B x 512 x 7 x 7
        B, dim, w, h = target_token.shape
        target_token = rearrange(target_token, 'b d h w -> b (h w) d', d = dim, w=w, h=h)
        target_token = self.target_linear(target_token)
        
        return target_token
        
    def encode_global(self, global_emb, position, pid=None, sid=None):
        # Global tokens
        if isinstance(global_emb, dict):
            global_token = torch.zeros((sid.shape[0], self.emb_dim)).to(sid.device)
            for _id, x_g in global_emb.items():
                batch_idx = pid == _id
                pos = position[_id]
                # x_cond_encoded = self.encode_cond(x_g, pos[id_]) # N x D
                g_token = self.global_encoder(x_g, pos).squeeze()  # N x 512
                global_token[batch_idx] = g_token[sid[batch_idx]] # B x D
        else:
            global_token = self.global_encoder(global_emb, position).squeeze()  # N x 512
            if sid is not None:
                global_token = global_token[sid]
                
        return global_token
        
    def _get_outputs(self, fusion_token, target_token, neighbor_token, global_token, label):
        output = self.fc(fusion_token) # B x num_genes
        out_target = self.fc_target(target_token.mean(1)) # B x num_genes
        out_neighbor = self.fc_neighbor(neighbor_token.mean(1)) # B x num_genes
        out_global = self.fc_global(global_token) # B x num_genes
        
        preds = (output, out_target, out_neighbor, out_global)
        
        loss = self.calculate_loss(preds, label)
        
        return {'loss': loss, 'logits': output}
        
    def calculate_loss(self, preds, label):
        
        loss = F.mse_loss(preds[0], label)                       # Supervised loss for Fusion
        
        for i in range(1, len(preds)):
            loss += F.mse_loss(preds[i], label) * (1-self.alpha) # Supervised loss
            loss += F.mse_loss(preds[0], preds[i]) * self.alpha  # Distillation loss
    
        return loss
    
    def retrieve_global_emb(self, pid, dataset):
        device = pid.device
        unique_pid = pid.unique()
        
        global_emb = {}
        pos = {}
        for pid in unique_pid:
            pid = int(pid)
            _id = dataset.int2id[pid]
            
            global_emb[pid] = dataset.global_embs[_id].clone().to(device).unsqueeze(0)
            pos[pid] = dataset.pos_dict[_id].clone().to(device)
        
        return global_emb, pos
    