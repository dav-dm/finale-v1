import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from module.head import FullyConnected
from network.base_network import BaseNetwork


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.register_buffer("pe", self._build_pe(max_len, d_model), persistent=False)

    @staticmethod
    def _build_pe(max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def _extend(self, new_len: int):
        if new_len > self.pe.size(1):
            self.pe = self._build_pe(new_len, self.d_model).to(self.pe.device, self.pe.dtype)

    def forward(self, x):
        """
        Args:
            x: tensor (B, L, d_model)
        """
        L = x.size(1)
        if L > self.pe.size(1):
            self._extend(L)

        pe = self.pe[:, :L].to(dtype=x.dtype, device=x.device)
        x = x * math.sqrt(self.d_model) + pe
        return self.dropout(x)


class BiflowTransformer(BaseNetwork):
    """
    Transformer model for 1D traffic classification.
    """
    def __init__(
        self, d_model=128, nhead=2, num_layers=1, dim_ff=256, dropout=0.1, **kwargs,
    ):
        super().__init__()
        
        num_classes = kwargs['num_classes']
        num_pkts = kwargs['num_pkts']
        num_fields = len(kwargs['fields'])
        self.out_features_size = 100
        # Token CLS
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Positional encoding
        pos_enc = PositionalEncoding(d_model, max_len=num_pkts + 1, dropout=dropout)
        # Stack encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True,
        )
        encoder = nn.TransformerEncoder(enc_layer, num_layers)

        # Backbone
        self.backbone = nn.ModuleDict({
            'in_proj': nn.Linear(num_fields, d_model),
            'pos_enc': pos_enc,
            'encoder': encoder,
            'fc1': nn.Linear(d_model, self.out_features_size),
        })
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        # Init the network with a FullyConnected head
        self.set_head(FullyConnected(in_features=self.out_features_size, num_classes=num_classes))

        self.apply(self._init_weights)
        
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
            

    def forward(self, x, return_feat=False):
        embeddings = self.extract_features(x)
        out = F.relu(embeddings) # Activate the embeddings
        out = self.head(out)
        if return_feat:
            return out, embeddings
        return out        
        
    def extract_features(self, x):
        # (B, 1, L, F) → (B, L, F)
        x = x.squeeze(1)
        B, _, _ = x.shape
        
        # Build pad mask 
        flag = x[..., 0]
        pad_mask = (flag == 0)  # (B, L) boolean
        pad_mask = torch.cat(
            [torch.zeros(B, 1, dtype=torch.bool, device=x.device), pad_mask], dim=1
        )  # (B, L+1) boolean
        
        out = self.backbone['in_proj'](x)       # (B, L, d_model)
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        out = torch.cat([cls, out], dim=1)      # (B, L+1, d_model)
        out = self.backbone['pos_enc'](out) 
        out = self.backbone['encoder'](out, src_key_padding_mask=pad_mask)  # (B, L+1, d_model)
        
        out = out[:, 0]  # (B, d_model)
        out = self.backbone['fc1'](out)
        return out