from torch import nn

from approach.dl_module import DLModule


class Scratch(DLModule):
    """
    Class for a deep learning module that includes training and validation
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ce_loss = nn.CrossEntropyLoss()
        self.net.summarize_module() if self.verbose else None
        self.configure_optimizers()
    
    
    def _fit_step(self, batch_x, batch_y):
        logits = self.net(batch_x)
        loss = self.ce_loss(logits, batch_y)
        return loss, logits
                
            
    def _predict_step(self, batch_x, batch_y):
        logits = self.net(batch_x)
        loss = self.ce_loss(logits, batch_y)
        return loss, logits
  