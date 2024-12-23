
from functools import partial

import torch.nn as nn
import torch.nn.functional as F


class FlashTransformerEncoder(nn.Module):
    def __init__(
        self,
        dim_model,
        num_layers,
        num_heads=None,
        dim_feedforward=None,
        dropout=0.0,
        norm_first=False,
        activation=F.gelu,
        rotary_emb_dim=0,
    ):
        super().__init__()

        try:
            from flash_attn.bert_padding import pad_input, unpad_input
            from flash_attn.modules.block import Block
            from flash_attn.modules.mha import MHA
            from flash_attn.modules.mlp import Mlp
        except ImportError:
            raise ImportError('Please install flash_attn from https://github.com/Dao-AILab/flash-attention')
        
        self._pad_input = pad_input
        self._unpad_input = unpad_input

        if num_heads is None:
            num_heads = dim_model // 64
        
        if dim_feedforward is None:
            dim_feedforward = dim_model * 4

        if isinstance(activation, str):
            activation = {
                'relu': F.relu,
                'gelu': F.gelu
            }.get(activation)

            if activation is None:
                raise ValueError(f'Unknown activation {activation}')

        mixer_cls = partial(
            MHA,
            num_heads=num_heads,
            use_flash_attn=True,
            rotary_emb_dim=rotary_emb_dim
        )

        mlp_cls = partial(Mlp, hidden_features=dim_feedforward)

        self.layers = nn.ModuleList([
            Block(
                dim_model,
                mixer_cls=mixer_cls,
                mlp_cls=mlp_cls,
                resid_dropout1=dropout,
                resid_dropout2=dropout,
                prenorm=norm_first,
            ) for _ in range(num_layers)
        ])
    
    def forward(self, x, src_key_padding_mask=None):
        # x = x.type(torch.float16)
        batch, seqlen = x.shape[:2]

        if src_key_padding_mask is None:
            for layer in self.layers:
                x = layer(x)[0]
        else:
            x, indices, cu_seqlens, max_seqlen_in_batch = self._unpad_input(x, ~src_key_padding_mask)
        
            for layer in self.layers:
                x = layer(x, mixer_kwargs=dict(
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen_in_batch
                ))[0]

            x = self._pad_input(x, indices, batch, seqlen)
            
        return x