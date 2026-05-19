import sys
import torch
from torch import nn

from approach.dl_module import DLModule
from module.maml_wrapper import MAMLWrapper
from util.config import load_config

disable_tqdm = not sys.stdout.isatty()


class MAML(DLModule):
    """
    MAML implements Model-Agnostic Meta-Learning for few-shot classification,
    as described in "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks."
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cf = load_config()

        self.task = 'src'
        self.classes_dict = self.num_classes

        self.ce_loss = nn.CrossEntropyLoss()
        self.num_ways = kwargs.get('num_ways', cf['num_ways'])
        self.train_k = kwargs.get('train_k', cf['train_k'])
        self.adapt_steps = kwargs.get('maml_adapt_steps', cf['maml_adapt_steps'])
        self.adapt_lr = kwargs.get('maml_adapt_lr', cf['maml_adapt_lr'])
        self.first_order = kwargs.get('maml_first_order', cf['maml_first_order'])

        self.maml_net = MAMLWrapper(
            self.net,
            lr=self.adapt_lr,
            first_order=self.first_order,
            allow_unused=True,
        )
        self.net.summarize_module() if self.verbose else None
        self._adapted_learner = None

        self.configure_optimizers(params=self.maml_net.parameters())


    @staticmethod
    def add_appr_specific_args(parent_parser):
        cf = load_config()
        parser = DLModule.add_appr_specific_args(parent_parser)
        parser.add_argument('--maml-adapt-steps', type=int, default=cf['maml_adapt_steps'])
        parser.add_argument('--maml-adapt-lr', type=float, default=cf['maml_adapt_lr'])
        parser.add_argument('--maml-first-order', action='store_true', default=cf['maml_first_order'])
        return parser


    @torch.enable_grad()
    def _fit_step(self, batch_x, batch_y):
        return self._meta_learn(batch_x, batch_y)
    

    @torch.enable_grad()
    def _predict_step(self, batch_x, batch_y):
        if self.task == 'src':
            return self._meta_learn(batch_x, batch_y)
        else:
            if self._adapted_learner is None:
                raise RuntimeError("Call adapt() before predict() on the target task.")
            logits = self._adapted_learner(batch_x)[self.task]
            loss = self.ce_loss(logits, batch_y)
            return loss, logits


    def _meta_learn(self, batch_x, batch_y):
        # Clone meta-learned weights, original self.net untouched
        learner = self.maml_net.clone()
        learner.train()

        (support_x, support_y), (query_x, query_y) = self.split_episode_batch(
            batch_x=batch_x, batch_y=batch_y,
        )
        support_y = support_y.long()
        query_y = query_y.long()

        # Inner loop -> adapt clone on support set
        for _ in range(self.adapt_steps):
            support_logits = learner(support_x)[self.task]
            support_loss = self.ce_loss(support_logits, support_y)
            learner.adapt(support_loss)

        # Outer loop -> evaluate adapted clone on query set
        logits = learner(query_x)[self.task]  # [num_query, num_classes]
        loss = self.ce_loss(logits, query_y)
        return loss, logits, query_y


    def _adapt(self, adapt_dataloader, val_dataloader, **kwargs):
        self._init_head('trg')

        learner = self.maml_net.clone()
        learner.train()

        # Collect all support examples
        xs, ys = [], []
        for batch_x, batch_y in adapt_dataloader:
            xs.append(batch_x.to(self.device))
            ys.append(batch_y.to(self.device).long())
        support_x = torch.cat(xs, dim=0)
        support_y = torch.cat(ys, dim=0)

        # Inner loop
        for _ in range(self.adapt_steps):
            logits = learner(support_x)[self.task]
            loss = self.ce_loss(logits, support_y)
            learner.adapt(loss)

        self._adapted_learner = learner

    
    def _init_head(self, who):
        # Xavier head initialization
        head = self.net.head.heads[who]
        for m in head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                        

    def set_task(self, task):
        if task not in ('src', 'trg'):
            raise ValueError("task must be either 'src' or 'trg'")
        self.task = task
        self.num_classes = self.classes_dict[self.task]