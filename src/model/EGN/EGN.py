
import os
import numpy as np

from scipy.stats import pearsonr
import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange,Reduce

import torch.nn.functional as F
import pytorch_lightning as pl

'''
code is based on https://github.com/lucidrains/vit-pytorch and https://github.com/Kevinz-code/CSRA

'''

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Projector(nn.Module):
    def __init__(
        self,
        mdim,
        dim,
        player,
        linear_projection,
        num_genes,
        dropout = 0.,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.proq = nn.Linear(mdim,dim) #for shape alignment
        self.proe = nn.Linear(mdim+num_genes,dim)
        for layer_num in range(player):
            self.layers.append(PreNorm(dim, FeedForward(dim, dim, dropout = dropout))) #use smooth version of relu, \ie, gelu
        self.to_v = nn.Linear(num_genes, dim) if linear_projection else \
                                                                    nn.Sequential(
                                                                    nn.Linear(num_genes, dim),
                                                                    nn.GELU(),
                                                                    nn.Linear(dim, dim),
                                                                )
                                                                
    def forward(self, q,oq,oq_v):
        exp = self.proe(torch.cat((oq,oq_v),-1))
        x = torch.cat((self.proq(q),exp),1)
        for ff in self.layers:
            x = ff(x) + x
        q,oq = x.split([q.size(1),exp.size(1)],1)
        return q,oq, self.to_v(oq_v)

class CSRA(nn.Module): 
    def __init__(self, dim, T = 0.1, lam = 0.5):
        super(CSRA, self).__init__()
        self.T = T           
        self.lam = lam                          
        self.head = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        score = self.head(x) 
        base_logit = score.mean(1)
        score_soft = self.softmax(score * self.T)
        att_logit = torch.sum(score * score_soft, dim=1)
        return base_logit + self.lam * att_logit

class Transformer(nn.Module):
    def __init__(self, mdim,dim,player, depth, linear_projection, heads, dim_head, mlp_dim, dropout,bhead,bdim,bfre, num_genes):
        super().__init__()
        self.encoder = Projector(mdim,dim,player, linear_projection, num_genes, dropout) 
        self.layers = nn.ModuleList([])
        for l in range(depth):
            cross_attn = True if (l + 1) % bfre == 0 else False
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)), 
                EB(dim, heads = bhead, dim_head = bdim, dropout = dropout) if cross_attn else None,
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)),
            ]))
        
        self.pool = CSRA(dim)
    def forward(self, x, data, pos):
        q = data["p_feature"]   
        oq = data["op_feature"]  
        oq_v = data["op_count"] 
        q,oq,oq_v = self.encoder(q,oq,oq_v)
        
        for attn, eb, ff in self.layers:
            x = attn(x + pos) + x
            if eb != None:
                x,q,oq = eb(x,q,oq,oq_v)
            x = ff(x) + x
        return torch.cat((self.pool(x),q.squeeze(1)),1)

class Update(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        
        self.query = nn.Sequential(
            nn.Linear(dim * 2, inner_dim),
            nn.SiLU(True),
            nn.Linear(inner_dim, dim * 2)
            )
        
        self.value = nn.Linear(dim, dim)
        # self.norm = nn.LayerNorm(dim)        
    def forward(self,q,k,v):
        value = self.value(v)
        query, key = self.query(torch.cat((q-k,k),2)).chunk(2,2)        
        return (query.sigmoid() * value).mean(1,True) + q, key.sigmoid() * value + k
    
class EB(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        h = 7
        w = 7
        
        self.update = Update(dim,heads = 4,dim_head=dim_head,dropout=dropout)

        self.to_x = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            Rearrange('b (h w) (n c) -> b n c h w ', n = heads, h=h, w = w),
            )
        
        self.to_k = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, h * w * heads),
            Rearrange('b l (n h w) -> b n l h w ', n = heads, h=h, w = w),
            ) 
            
        self.to_v = nn.Sequential(
            Rearrange('b n c h w -> b (h w) (n c)'),
            nn.Linear(inner_dim, dim, bias = False)
            )

        
        self.to_n = nn.Sequential(
            Reduce('(b l) n c h w -> b l (n c)', 'mean',l = 1),
            nn.Linear(inner_dim, dim, bias = False)
            )

    def forward(self, x, q,oq,oq_v):
        new_q, new_k = self.update(q,oq,oq_v)
        key = self.to_k(new_q)
        value = self.to_x(x) * key.sigmoid()
        x = self.to_v(value) + x 
        q = self.to_n(value) + q
        return x,q,new_k

def pearson_R(x, y):
    
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / (r_den + 1e-8)
    r_val = torch.nan_to_num(r_val,nan=-1)
    return r_val

class EGN(nn.Module):
    def __init__(self, bhead=16, bdim=128, bfre=2, mdim=1024, player=2, linear_projection=True,
                image_size = 224, patch_size = 32, num_outputs = 300, dim = 1024, 
                depth = 16, heads = 16, mlp_dim = 2048, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.,
                max_batch_size = 1024):
        super().__init__()
        
        self.dim = dim
        self.max_batch_size = max_batch_size

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(mdim,dim, player, depth,linear_projection, heads, dim_head, mlp_dim, dropout,bhead,bdim,bfre, num_outputs)
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim * 2),
                nn.Linear(dim * 2, num_outputs)
            )
            
    def forward(self, img, ei, ej, yj, label=None, **kwargs):
        phase = kwargs.get('phase', 'train')
        
        if phase == 'train':
            output = self._forward_single(img, ei, ej, yj)
            
        else:
            if img.shape[0] > self.max_batch_size:
                imgs = img.split(self.max_batch_size, dim=0)
                ei = ei.split(self.max_batch_size, dim=0)
                ej = ej.split(self.max_batch_size, dim=0)
                yj = yj.split(self.max_batch_size, dim=0)
                output = [self._forward_single(imgs[i], ei[i], ej[i], yj[i]) for i in range(len(imgs))]
                output = torch.cat(output, dim=0)
            else:
                output = self._forward_single(img, ei, ej, yj)
        
        output = torch.clamp(output, 0) 
            
        if label is not None:
            loss = F.mse_loss(output, label)
            corrloss = self.correlationMetric(output, label)
            loss = loss + corrloss * 0.5
            
            result_dict = {'loss': loss, 'logits': output}
        else:
            result_dict = {'logits': output}
        
        return result_dict
    
    def _forward_single(self, img, ei, ej, yj):
        data = {'img': img, 'p_feature':ei,'op_feature':ej, 'op_count':yj}
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        pos = self.pos_embedding[:, :n]
        x = self.dropout(x + pos)
        x = self.transformer(x, data, pos)
        x = self.to_latent(x)
        output = self.mlp_head(x)
        
        return output
    
    def correlationMetric(self,x, y):
        corr = 0
        for idx in range(x.size(1)):
            corr += pearson_R(x[:,idx], y[:,idx])
        corr /= (idx + 1)
        return (1 - corr).mean()


if __name__ == '__main__':
    m = EGN(
            image_size = 256,
            dim = 1024,
            depth = 8,
            heads = 16,
            mlp_dim = 4096,
            bhead = 8,
            bdim = 64,
            bfre = 2,
            mdim=1024,
            player = 1, 
            linear_projection = True,
        )    

    data = dict(
        img = torch.randn((8,3,224,224)),
        p_feature = torch.randn((8,1,1024)),
        op_feature = torch.randn((8,9,1024)),
        op_count = torch.randn((8,9,300)),
        )
    
    with torch.no_grad():
        y = m(data)
        print(y.shape)
