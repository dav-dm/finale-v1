from torch import nn

from approach.dl_module import DLModule
from module.head import MultiHead
from module.negative_margin_layer import NegativeMarginLayer
from util.config import load_config


class NegativeMargin(DLModule):
    """
    [[Link to Source Code]](https://github.com/RL-VIG/LibFewShot)
    NegativeMargin is a class that implements the Negative Margin Matters algorithm 
    for few-shot learning, as described in "Negative Margin Matters: Understanding Margin in
    Few-shot Classification".
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cf = load_config()
        
        self.task = 'src'
        self.classes_dict = self.num_classes
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.margin = kwargs.get('nm_margin', cf['nm_margin'])
        self.inner_margin = kwargs.get('nm_inner_margin', cf['nm_inner_margin'])
        self.temp = kwargs.get('nm_temp', cf['nm_temp'])
        self.inner_temp = kwargs.get('nm_inner_temp', cf['nm_inner_temp'])
        self.adapt_lr = kwargs.get('adapt_lr', cf['adapt_lr'])
        self.adapt_epochs = kwargs.get('adapt_epochs', cf['adapt_epochs'])
        
        self.net.set_head(
            MultiHead({
                'src': NegativeMarginLayer(
                    self.net.out_features_size, self.classes_dict['src'], 
                    margin=self.margin, temperature=self.temp).to(self.device),
                'trg': NegativeMarginLayer(
                    self.net.out_features_size, self.classes_dict['trg'],
                    margin=self.inner_margin, temperature=self.inner_temp).to(self.device),
            })
        )
        self.net.summarize_module() if self.verbose else None
        
        self.configure_optimizers()
        
        
    @staticmethod
    def add_appr_specific_args(parent_parser):
        cf = load_config()
        parser = DLModule.add_appr_specific_args(parent_parser)
        parser.add_argument('--nm-margin', type=float, default=cf['nm_margin'])
        parser.add_argument('--nm-inner-margin', type=float, default=cf['nm_inner_margin'])
        parser.add_argument('--nm-temp', type=float, default=cf['nm_temp'])
        parser.add_argument('--nm-inner-temp', type=float, default=cf['nm_inner_temp'])
        return parser
    
    
    def _fit_step(self, batch_x, batch_y):
        _, batch_emb = self.net(batch_x, return_feat=True)
        logits = self.net.head.heads[self.task](batch_emb, batch_y)
        loss = self.ce_loss(logits, batch_y)
        return loss, logits
    
    
    def _predict_step(self, batch_x, batch_y):
        # NegativeMarginLayer does not use label during inference, 
        # so we can directly call self.net(batch_x)[self.task]
        logits = self.net(batch_x)[self.task]
        loss = self.ce_loss(logits, batch_y) 
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