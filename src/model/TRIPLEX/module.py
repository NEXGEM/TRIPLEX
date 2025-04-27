

import itertools

import torch
from torch import nn
from einops import rearrange

from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
try:
    import MinkowskiEngine as ME
    HAS_MINKOWSKI = True
except ImportError:
    HAS_MINKOWSKI = False


class PreNorm(nn.Module):
    def __init__(self, emb_dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        x = self.norm(x)
        if 'x_kv' in kwargs.keys():
            kwargs['x_kv'] = self.norm(kwargs['x_kv'])
         
        return self.fn(x, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, emb_dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, heads = 4, dropout = 0., attn_bias=False, resolution=(5, 5), flash_attn=False):
        super().__init__()
        
        assert emb_dim % heads == 0, 'The dimension size must be a multiple of the number of heads.'
        
        self.flash_attn = flash_attn
        
        dim_head = emb_dim // heads 
        project_out = not (heads == 1) 

        self.heads = heads
        self.drop_p = dropout
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        
        self.to_qkv = nn.Linear(emb_dim, emb_dim * 3, bias = False) 

        self.to_out = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
        self.attn_bias = attn_bias
        if attn_bias:
            points = list(itertools.product(
                range(resolution[0]), range(resolution[1])))
            N = len(points)
            attention_offsets = {}
            idxs = []
            for p1 in points:
                for p2 in points:
                    offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                    if offset not in attention_offsets:
                        attention_offsets[offset] = len(attention_offsets)
                    idxs.append(attention_offsets[offset])
            self.attention_biases = torch.nn.Parameter(
                torch.zeros(heads, len(attention_offsets)))
            self.register_buffer('attention_bias_idxs',
                                torch.LongTensor(idxs).view(N, N),
                                persistent=False)

    @torch.no_grad()
    def train(self, mode=True):
        if self.attn_bias:
            super().train(mode)
            if mode and hasattr(self, 'ab'):
                del self.ab
            else:
                self.ab = self.attention_biases[:, self.attention_bias_idxs]
        
    def forward(self, x, mask = None, return_attn=False):
        
        qkv = self.to_qkv(x) # b x n x d*3
        
        if self.flash_attn:
            qkv = rearrange(qkv, 'b n (h d a) -> b n a h d', h = self.heads, a=3)
            out = flash_attn_qkvpacked_func(qkv, self.drop_p, softmax_scale=None, causal=False)
            out = rearrange(out, 'b n h d -> b n (h d)')
            
        else:
            qkv = qkv.chunk(3, dim = -1) 
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv) 

            qk = torch.matmul(q, k.transpose(-1, -2)) * self.scale 
            if self.attn_bias:
                qk += (self.attention_biases[:, self.attention_bias_idxs]
                if self.training else self.ab)
            
            if mask is not None:
                fill_value = torch.finfo(torch.float16).min
                ind_mask = mask.shape[-1]
                qk[:,:,-ind_mask:,-ind_mask:] = qk[:,:,-ind_mask:,-ind_mask:].masked_fill(mask==0, fill_value)

            attn_weights = self.attend(qk) # b h n n
            if return_attn:
                attn_weights_averaged = attn_weights.mean(dim=1)
            
            out = torch.matmul(attn_weights, v) 
            out = rearrange(out, 'b h n d -> b n (h d)')
        
            if return_attn:
                return self.to_out(out), attn_weights_averaged[:,0]
            
        return self.to_out(out)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, emb_dim, heads = 4, dropout = 0., flash_attn=False):
        super().__init__()
        
        assert emb_dim % heads == 0, 'The dimension size must be a multiple of the number of heads.'
        
        self.flash_attn = flash_attn
        
        dim_head = emb_dim // heads 
        project_out = not (heads == 1) 

        self.heads = heads
        self.drop_p = dropout
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        
        self.to_q = nn.Linear(emb_dim, emb_dim, bias = False) 
        self.to_kv = nn.Linear(emb_dim, emb_dim * 2, bias = False) 

        self.to_out = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
    def forward(self, x_q, x_kv, mask = None, return_attn=False):
        q = self.to_q(x_q)
        kv = self.to_kv(x_kv).chunk(2, dim = -1) 
        
        if self.flash_attn:
            q = rearrange(q, 'b n (h d) -> b n h d', h = self.heads)
            k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h = self.heads), kv) 
            
            out = flash_attn_func(q, k, v)
            out = rearrange(out, 'b n h d -> b n (h d)')
            
        else:
            q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
            k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), kv) 

            qk = torch.matmul(q, k.transpose(-1, -2)) * self.scale 
            
            if mask is not None:
                fill_value = torch.finfo(torch.float16).min
                ind_mask = mask.shape[-1]
                qk[:,:,-ind_mask:,-ind_mask:] = qk[:,:,-ind_mask:,-ind_mask:].masked_fill(mask==0, fill_value)

            attn_weights = self.attend(qk) # b h n n
            if return_attn:
                attn_weights_averaged = attn_weights.mean(dim=1)
            
            out = torch.matmul(attn_weights, v) 
            out = rearrange(out, 'b h n d -> b n (h d)')
        
            if return_attn:
                return self.to_out(out), attn_weights_averaged[:,0]
            
        return self.to_out(out)


class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout = 0., attn_bias=False, resolution=(5,5), flash_attn=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(emb_dim, MultiHeadAttention(emb_dim, heads = heads, 
                                                    dropout = dropout, attn_bias=attn_bias, 
                                                    resolution=resolution, flash_attn=flash_attn)),
                PreNorm(emb_dim, FeedForward(emb_dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, mask=None, return_attn=False):
        for attn, ff in self.layers:
            if return_attn:
                attn_out, attn_weights = attn(x, mask=mask, return_attn=return_attn)
                x += attn_out # residual connection after attention      
                x = ff(x) + x # residual connection after feed forward net
                
            else:
                x = attn(x, mask=mask) + x # residual connection after attention      
                x = ff(x) + x # residual connection after feed forward net
            
        if return_attn:
            return x, attn_weights
        else:
            return x


class CrossEncoder(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(emb_dim, MultiHeadCrossAttention(emb_dim, heads = heads, dropout = dropout)),
                PreNorm(emb_dim, FeedForward(emb_dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x_q, x_kv, mask=None, return_attn=False):
        for attn, ff in self.layers:
            if return_attn:
                attn_out, attn_weights = attn(x_q, x_kv=x_kv, mask=mask, return_attn=return_attn)
                x_q += attn_out # residual connection after attention      
                x_q = ff(x_q) + x_q # residual connection after feed forward net
            else:
                x_q = attn(x_q, x_kv=x_kv, mask=mask) + x_q
                x_q = ff(x_q) + x_q # residual connection after feed forward net

        if return_attn:
            return x_q, attn_weights
        else:
            return x_q
        
        
class APEG(nn.Module):
    def __init__(self, dim=512, kernel_size=3, grid_size=None, use_sparse=False, sparse_resolution=128):
        super(APEG, self).__init__()
        self.use_sparse = use_sparse
        self.grid_size = grid_size  # (H, W) or None
        self.sparse_resolution = sparse_resolution

        if self.use_sparse:
            if not HAS_MINKOWSKI:
                raise ImportError("MinkowskiEngine is not installed. Install it via 'pip install MinkowskiEngine'.")
            self.proj = ME.MinkowskiConvolution(
                in_channels=dim,
                out_channels=dim,
                kernel_size=kernel_size,
                stride=1,
                dimension=2,
                bias=True
            )
        else:
            self.proj = nn.Conv2d(
                dim, dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=True,
                groups=dim  # depthwise
            )

    def infer_grid_size(self, pos, rounding_factor=None):
        """
        pos: (N, 2) tensor
        """
        if rounding_factor is None:
            rounding_factor = self.dynamic_rounding_factor(pos)
        
        pos_rounded = (pos / rounding_factor).round() * rounding_factor
        unique_x = torch.unique(pos_rounded[:, 0])
        unique_y = torch.unique(pos_rounded[:, 1])
        W = unique_x.numel()
        H = unique_y.numel()
        return (W, H)
    
    @staticmethod
    def dynamic_rounding_factor(pos, base=1, scale_ref=1000.0):
        """
        Args:
            pos: (N, 2) tensor
            base: base rounding factor (based on scale_ref)
            scale_ref: base pos range (range: 1000 -> 1 )
        Returns:
            rounding_factor: float
        """
        pos_range = (pos.max(0)[0] - pos.min(0)[0]).max().item()  # x, y 중 큰 축
        scale_ratio = pos_range / scale_ref
        rounding_factor = base * scale_ratio
        return rounding_factor
    
    def forward(self, x, pos):
        """
        Args:
            x: (1, N, dim)
            pos: (N, 2)
        Returns:
            x_out: (1, N, dim)
        """
        B, N, C = x.shape
        device = x.device

        if self.use_sparse:
            # --- Sparse Version ---
            pos_min = pos.min(dim=0, keepdim=True)[0]
            pos_max = pos.max(dim=0, keepdim=True)[0]
            pos_norm = (pos - pos_min) / (pos_max - pos_min + 1e-5)

            discrete_coords = (pos_norm * (self.sparse_resolution - 1)).round().int()

            batch_indices = torch.zeros((N, 1), dtype=torch.int, device=device)
            coords = torch.cat([batch_indices, discrete_coords], dim=1)  # (N, 1+2)

            sparse_input = ME.SparseTensor(
                features=x.squeeze(0),
                coordinates=coords,
            )

            sparse_output = self.proj(sparse_input)

            # Matching input coords and output coords
            out_features = sparse_output.features
            out_coords = sparse_output.coordinates[:, 1:]

            match_idx = []
            for i in range(N):
                match = ((out_coords == discrete_coords[i]).all(dim=1)).nonzero(as_tuple=True)[0]
                match_idx.append(match.item())

            match_idx = torch.tensor(match_idx, device=device, dtype=torch.long)
            matched_features = out_features[match_idx]

            x_out = matched_features.unsqueeze(0)
            return x_out

        else:
            # --- Dense Version ---
            if self.grid_size is None:
                self.grid_size = self.infer_grid_size(pos, rounding_factor=20)  # (W, H) inferred

            W, H = self.grid_size

            pos_min = pos.min(dim=0, keepdim=True)[0]
            pos_max = pos.max(dim=0, keepdim=True)[0]
            pos_norm = (pos - pos_min) / (pos_max - pos_min + 1e-5)
            grid_pos = pos_norm * torch.tensor([W - 1, H - 1], device=device)
            grid_pos = grid_pos.round().long()

            gx, gy = grid_pos[:, 0], grid_pos[:, 1]  # (N,), (N,)

            # Flatten 2D grid into 1D index for scatter
            idx_1d = gy * W + gx  # (N,)

            dense_x = torch.zeros((B, C, H * W), device=device)
            dense_mask = torch.zeros((B, 1, H * W), device=device)

            # Scatter add (optimized)
            for b in range(B):
                dense_x[b] = dense_x[b].scatter_add(1, idx_1d.unsqueeze(0).expand(C, -1), x[b].transpose(0,1))
                dense_mask[b] = dense_mask[b].scatter_add(1, idx_1d.unsqueeze(0), torch.ones(1, N, device=device))

            dense_x = dense_x.view(B, C, H, W)
            dense_mask = dense_mask.view(B, 1, H, W)

            dense_mask = dense_mask.clamp(min=1.0)
            dense_x = dense_x / dense_mask

            x_pos = self.proj(dense_x)

            mask = (dense_mask > 0)
            x_pos = x_pos * mask + dense_x * (~mask)

            # Sampling
            x_out = []
            for b in range(B):
                sampled_feat = x_pos[b, :, grid_pos[:, 1], grid_pos[:, 0]].transpose(0,1)  # (N, C)
                x_out.append(sampled_feat)

            x_out = torch.stack(x_out, dim=0)  # (B, N, C)
            return x_out
        
    
class GlobalEncoder(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout = 0., kernel_size=3):
        super().__init__()      
        
        self.pos_layer = APEG(dim=emb_dim, kernel_size=kernel_size) 
        
        self.layer1 = TransformerEncoder(emb_dim, 1, heads, mlp_dim, dropout, flash_attn=True)
        self.layer2 = TransformerEncoder(emb_dim, depth-1, heads, mlp_dim, dropout, flash_attn=True)
        self.norm = nn.LayerNorm(emb_dim)
        
    def foward_features(self, x, pos):
        # Translayer x1
        x = self.layer1(x) #[B, N, 384]

        # PEGH
        x = self.pos_layer(x, pos) #[B, N, 384]        
        
        # Translayer x (depth-1)
        x = self.layer2(x) #[B, N, 384]        
        x = self.norm(x) 
        
        return x
        
    def forward(self, x, position):    
        x = self.foward_features(x, position) # 1 x N x 384
    
        return x
    
    
class NeighborEncoder(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout = 0., resolution=(5,5)):
        super().__init__()      
        
        self.layer = TransformerEncoder(emb_dim, depth, heads, mlp_dim, dropout, attn_bias=True, resolution=resolution)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x, mask=None):
        
        if mask != None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            
        # Translayer
        x = self.layer(x, mask=mask) #[B, N, 512]
        x = self.norm(x)
        
        return x


class FusionEncoder(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout):
        super().__init__()      
                
        self.fusion_layer = CrossEncoder(emb_dim, depth, heads, mlp_dim, dropout)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x_t=None, x_n=None, x_g=None, mask=None):
            
        if mask != None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            
        # Target token
        fus1 = self.fusion_layer(x_g.unsqueeze(1), x_t) 
            
        # Neighbor token
        fus2 = self.fusion_layer(x_g.unsqueeze(1), x_n, mask=mask) 
                
        fusion = (fus1 + fus2).squeeze(1)            
        fusion = self.norm(fusion)
        
        return fusion
