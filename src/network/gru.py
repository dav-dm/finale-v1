import math
import torch
import torch.nn.functional as F
from torch import nn

from module.head import FullyConnected
from network.base_network import BaseNetwork
from util.config import load_config

class GRU(BaseNetwork):
    """
    GRU model from "Cross-Evaluation of Deep Learning-Based Network Intrusion Detection Systems"
    """
    def __init__(self, in_channels=1, **kwargs):
        super().__init__()

        num_classes = kwargs['num_classes']
        num_pkts = kwargs['num_pkts']
        num_fields = len(kwargs['fields'])
        self.out_features_size = 256
        
        cf = load_config()
        scaling_factor = cf['net_scale']
        filter = math.ceil(64 * scaling_factor)
        
        # Backbone        
        self.backbone = nn.ModuleDict({
            'rnn': nn.GRU(num_fields, filter, batch_first=True, bidirectional=True),
            'dropout': nn.Dropout(p=0.2),
            'fc1': nn.Linear(num_pkts * 2 * filter, self.out_features_size),
        })
        # Init the network with a FullyConnected head
        self.set_head(FullyConnected(in_features=self.out_features_size, num_classes=num_classes))
        
        
    def forward(self, x, return_feat=False):
        embeddings = self.extract_features(x)
        out = F.relu(embeddings) # Activate the embeddings
        out = self.head(out)
        if return_feat:
            return out, embeddings
        return out
    
    def extract_features(self, x):
        # (B, 1, L, F) → (B, L, F)
        out = x.squeeze(1)

        self.backbone['rnn'].flatten_parameters()
        out, _ = self.backbone['rnn'](out)
        out = F.relu(out)
        
        out = torch.flatten(out, start_dim=1)
        out = self.backbone['dropout'](out)
        out = self.backbone['fc1'](out)
        return out
            
 