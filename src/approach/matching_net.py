import sys
import torch
from torch import nn
from tqdm import tqdm

from approach.dl_module import DLModule
from module.head import MatchingHead
from util.config import load_config

disable_tqdm = not sys.stdout.isatty()


class MatchingNet(DLModule):
    """
    MatchingNet is a class that implements the Matching Networks algorithm 
    for few-shot classification, as described in "Matching Networks for One Shot Learning" 
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cf = load_config()

        self.task = 'src'
        self.classes_dict = self.num_classes

        self.nll_loss = nn.NLLLoss()
        self.num_ways = kwargs.get('num_ways', cf['num_ways'])
        self.train_k = kwargs.get('train_k', cf['train_k'])

        self.net.set_head(
            MatchingHead().to(self.device)
        )
        self.net.summarize_module() if self.verbose else None

        self.configure_optimizers()

    
    def _fit_step(self, batch_x, batch_y):
        return self._meta_learn(batch_x, batch_y)


    def _predict_step(self, batch_x, batch_y):
        if self.task == 'src':
             # During meta-validation and meta-testing on the source (non-few shot) dataset
             return self._meta_learn(batch_x, batch_y)
        else:
            logits = self.net(batch_x)
            loss = self.nll_loss(logits, batch_y)
            return loss, logits


    def _meta_learn(self, batch_x, batch_y):
        (support_x, support_y), (query_x, query_y) = self.split_episode_batch(
            batch_x=batch_x,
            batch_y=batch_y,
        )
        support_y = support_y.long()
        query_y = query_y.long()

        # Encode labels from 0 to num_ways-1
        episode_classes, support_y_local = self.global_labels_to_local_labels(
            global_y=support_y
        )

        # Embed the input
        _, emb_support = self.net(support_x, return_feat=True)
        _, emb_query = self.net(query_x, return_feat=True)

        # Fit the NN head with support set
        self.net.head.fit(emb_support, support_y_local)

        # Predict query labels
        local_logits = self.net.head(emb_query)
        global_logits = self.local_logits_to_global_logits(
            local_logits=local_logits,
            episode_classes=episode_classes,
        )
        loss = self.nll_loss(global_logits, query_y)
        return loss, global_logits, query_y


    @torch.no_grad()
    def _adapt(self, adapt_dataloader, val_dataloader, **kwargs):
        self.net.freeze_backbone()  
        # self.net.trainability_info()
        self.net.eval()

        embeddings, labels = [], []

        adapt_loop = tqdm(
            adapt_dataloader, desc='[fitting Matching head]', 
            leave=True, disable=disable_tqdm
        ) if self.verbose else adapt_dataloader
        for batch_x, batch_y in adapt_loop:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device).long()
            
            # Embed the input
            _, batch_emb = self.net(batch_x, return_feat=True)
            embeddings.append(batch_emb)
            labels.append(batch_y)

        # Store train embedding and labels
        self.net.head.fit(x=torch.cat(embeddings), y=torch.cat(labels))


    def set_task(self, task):
        if task not in ['src', 'trg']:
            raise ValueError("Task should be either 'src' or 'trg'")
        self.task = task
        self.num_classes = self.classes_dict[self.task]