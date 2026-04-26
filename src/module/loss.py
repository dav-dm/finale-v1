import torch
from torch import nn


class DistillKLLoss(nn.Module):
    """
    Implementation of the Kullback-Leibler divergence for distilliation
    """
    def __init__(self, T):
        super(DistillKLLoss, self).__init__()
        self.T = T

    def forward(self, y_s, y_t, already_soft_values=False):
        if y_t is None:
            return 0.0

        p_s = nn.functional.log_softmax(y_s / self.T, dim=1)
        p_t = nn.functional.softmax(y_t / self.T, dim=1) if not already_soft_values else y_t
        loss = nn.functional.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.size(0)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss for multi-class classification.
        Args:
            alpha (float or list of floats, optional): Weighting factor for each class. 
                  If None, all classes are equally weighted.
            gamma (float): Focusing parameter. Default is 2.0.
            reduction (str): Reduction method. Options are 'mean', 'sum'.
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer('alpha', torch.as_tensor(alpha, dtype=torch.float))
        else:
            self.alpha = None

    def forward(self, logits, labels):
        log_pt = -nn.functional.cross_entropy(logits, labels, reduction='none')
        pt = log_pt.exp()
        
        # Compute the focal loss, forumulated as: 
        # FL(p_t) = - (1 - p_t)^gamma * log(p_t)
        loss = (1 - pt).pow(self.gamma) * (-log_pt)
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            if self.alpha.ndim == 0: # alpha is a scalar
                alpha_t = torch.where(labels==0, 1-self.alpha, self.alpha)
            else:                    # alpha is a tensor
                alpha_t = self.alpha[labels]
            loss *= alpha_t

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
     