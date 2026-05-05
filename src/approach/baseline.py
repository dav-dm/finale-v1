from torch import nn

from approach.dl_module import DLModule
from util.config import load_config


class Baseline(DLModule):
    """
    Baseline class for a deep learning module that includes training, validation, 
    and adaptation strategies (finetuning or freezing).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cf = load_config()
        
        self.task = 'src'
        self.classes_dict = self.num_classes
        
        self.criterion = nn.CrossEntropyLoss()
        self.adapt_strat = kwargs.get('adapt_strat', cf['adapt_strat'])
        self.adapt_lr = kwargs.get('adapt_lr', cf['adapt_lr'])
        self.adapt_epochs = kwargs.get('adapt_epochs', cf['adapt_epochs'])

        self.net.summarize_module() if self.verbose else None
        
        self.configure_optimizers()
        
        
    @staticmethod
    def add_appr_specific_args(parent_parser):
        cf = load_config()
        parser = DLModule.add_appr_specific_args(parent_parser)
        parser.add_argument('--adapt-strat', type=str, default=cf['adapt_strat'], 
                            choices=['finetuning', 'freezing'])
        parser.add_argument('--adapt-lr', type=float, default=cf['adapt_lr'])
        parser.add_argument('--adapt-epochs', type=int, default=cf['adapt_epochs'])
        return parser
    
    
    def _fit_step(self, batch_x, batch_y):
        logits = self.net(batch_x)[self.task]
        loss = self.criterion(logits, batch_y)
        return loss, logits
                
            
    def _predict_step(self, batch_x, batch_y):
        logits = self.net(batch_x)[self.task]
        loss = self.criterion(logits, batch_y)
        return loss, logits
    
        
    def _adapt(self, adapt_dataloader, val_dataloader):
        if self.adapt_strat == 'freezing':
            # Freeze the backbone
            self.net.freeze_backbone()
            # self.net.trainability_info()
        
        # Update hyperparameters for adaptation
        self.max_epochs = self.adapt_epochs
        self.lr = self.adapt_lr
        # Setup the the optimizer, if adapt_strat is finetuning params = self.net.parameters()
        params = self.net.head.parameters() if self.adapt_strat == 'freezing' else None
        self.configure_optimizers(params=params)

        self._fit(adapt_dataloader, val_dataloader)
        
        
    def set_task(self, task):
        if task not in ['src', 'trg']:
            raise ValueError("Task should be either 'src' or 'trg'")
        self.task = task
        self.num_classes = self.classes_dict[self.task]
        