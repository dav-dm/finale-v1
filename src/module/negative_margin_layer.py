import math
import torch
import torch.nn.functional as F
from torch import nn


class NegativeMarginLayer(nn.Module):
    """
    From "Negative Margin Matters: Understanding Margin in Few-shot Classification"
    Implemented from: https://github.com/RL-VIG/LibFewShot
    """
    def __init__(self, in_features, out_features, margin=-0.3, temperature=30.0):
        super(NegativeMarginLayer, self).__init__()
        self.margin = margin
        self.temperature = temperature
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, feature, label=None):
        device = feature.device
        cosine = F.linear(F.normalize(feature), F.normalize(self.weight.to(device)))
        # when test, no label, just return
        if label is None:
            return cosine * self.temperature

        phi = cosine - self.margin

        output = torch.where(self.one_hot(label, cosine.shape[1]).bool(), phi, cosine)
        output *= self.temperature
        return output

    def one_hot(self, y, num_class):
        return (
            torch.zeros((len(y), num_class)).to(y.device).scatter_(1, y.unsqueeze(1), 1)
        )