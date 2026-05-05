from tqdm import tqdm

from approach.approach_factory import get_approach
from trainer.base_trainer import BaseTrainer


class FewShotLearningTrainer(BaseTrainer):
    """
    Trainer for transfer learning.
    It first trains on the source dataset, then adapts to the target dataset.
    """
    def __init__(self, args, datasets):
        super().__init__(args, datasets)

    def run(self):
        # Only two datasets are expected: source (first one) and target (second one)
        items = list(self.datasets.items())
        (src_dataset_name, (src_splits, src_datamodule)) = items[0]
        (trg_dataset_name, (trg_splits, trg_datamodule)) = items[1]
        
        self.dict_args['num_classes'] = {
            'src': src_splits.num_classes,
            'trg': trg_splits.num_classes
        }
        self.dict_args['src_dataset'] = src_dataset_name
        self.dict_args['trg_dataset'] = trg_dataset_name
        
        self.dm.update_log_dir(src_dataset_name)
        self._save_dict_args()
        
        approach = get_approach(approach_name=self.args.approach, **self.dict_args)
        print('='*100)
        
        # Train, val, and test on src
        approach.datamodule = src_datamodule
        print(f'[Trainer] Starting training on source dataset: {src_dataset_name}')
        approach.set_task('src')
        approach.fit()
        approach.validate()
        approach.test()
        
        checkpoint_path = self.args.ckpt_path or approach.checkpoint_path
        
        self.dm.update_log_dir(trg_dataset_name) # Switch the log_dir to trg
        
        # Eisodic adaptation to trg
        print(
            f'[Trainer] Starting adaptation on target dataset: {trg_dataset_name} '
            f'for {self.args.num_episodes} episodes'
        )
        adapt_loop = tqdm(range(self.args.num_episodes), desc=f'Episodes')
        for episode_idx in adapt_loop:
            approach = self._reset_approach(checkpoint_path)
            approach.datamodule = trg_datamodule
            approach.set_task('trg')

            approach.adapt(episode_idx)
            approach.test(episode_idx)

            adapt_loop.set_postfix({'trn f1': f'{approach.outputs["f1_score_macro"]:.4f}'})
   
    def _reset_approach(self, checkpoint_path):        
        approach = get_approach(approach_name=self.args.approach, verbose=False, **self.dict_args)
        # Loads checkpoint from the first task or from checkpoint_path
        approach.load_checkpoint(checkpoint_path) 
        # print(f'[Trainer] Loaded approach state from {checkpoint_path}')
        return approach    
            
            