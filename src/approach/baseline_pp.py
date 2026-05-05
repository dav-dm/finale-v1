from torch import nn

from approach.dl_module import DLModule
from module.dist_linear import DistLinear
from module.head import MultiHead
from util.config import load_config


class BaselinePP(DLModule):
    """
    [[Link to Source Code]](https://github.com/RL-VIG/LibFewShot)
    BaselinePP is a class that implements the Baseline++ algorithm for few-shot classification,
    as described in "A Closer Look at Few-shot Classification. ICLR 2019."
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cf = load_config()
        
        self.task = 'src'
        self.classes_dict = self.num_classes
        
        self.criterion = nn.CrossEntropyLoss()
        self.adapt_lr = kwargs.get('adapt_lr', cf['adapt_lr'])
        self.adapt_epochs = kwargs.get('adapt_epochs', cf['adapt_epochs'])
        
        self.net.set_head(
            MultiHead({
                'src': DistLinear(self.net.out_features_size, self.classes_dict['src']).to(self.device),
                'trg': DistLinear(self.net.out_features_size, self.classes_dict['trg']).to(self.device),
            })
        )
    
    
    def _fit_step(self, batch_x, batch_y):
        logits = self.net(batch_x)[self.task]
        loss = self.criterion(logits, batch_y)
        return loss, logits
                
            
    def _predict_step(self, batch_x, batch_y):
        logits = self.net(batch_x)[self.task]
        loss = self.criterion(logits, batch_y)
        return loss, logits
    
        
    def _adapt(self, adapt_dataloader, val_dataloader):
        # Freeze the backbone
        self.net.freeze_backbone()
        # self.net.trainability_info()
        
        # Update hyperparameters for adaptation
        self.max_epochs = self.adapt_epochs
        self.lr = self.adapt_lr
        # Setup the the optimizer
        params = self.net.head.parameters()
        self.configure_optimizers(params=params)

        self._fit(adapt_dataloader, val_dataloader)
        
        
    def set_task(self, task):
        if task not in ['src', 'trg']:
            raise ValueError("Task should be either 'src' or 'trg'")
        self.task = task
        self.num_classes = self.classes_dict[self.task]
        