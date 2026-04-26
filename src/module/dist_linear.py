import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


class DistLinear(nn.Module):
    """
    From "A Closer Look at Few-shot Classification. ICLR 2019."
    Implemented from: https://github.com/RL-VIG/LibFewShot
    """
    def __init__(self, in_channel, out_channel):
        super(DistLinear, self).__init__()
        self.fc = nn.Linear(in_channel, out_channel, bias=False)

        self.class_wise_learnable_norm = True
        if self.class_wise_learnable_norm:
            weight_norm(self.fc, name="weight", dim=0)

        self.scale_factor = 2 if out_channel <= 200 else 10

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)

        if not self.class_wise_learnable_norm:
            fc_norm = (
                torch.norm(self.fc.weight.data, p=2, dim=1)
                .unsqueeze(1)
                .expand_as(self.fc.weight.data)
            )
            self.fc.weight.data = self.fc.weight.data.div(fc_norm + 0.00001)

        cos_dist = self.fc(x_normalized)
        score = self.scale_factor * cos_dist

        return score