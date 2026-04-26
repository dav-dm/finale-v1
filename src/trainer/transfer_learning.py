from approach.approach_factory import get_approach
from trainer.base_trainer import BaseTrainer


class TransferLearningTrainer(BaseTrainer):
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
        
        self.dict_args['num_classes'] = trg_splits.num_classes
        
        self.dm.update_log_dir(src_dataset_name)
        self._save_dict_args()
        
        approach = get_approach(approach_name=self.args.approach, **self.dict_args)
        print('='*100)
        
        # Train and val on src
        approach.datamodule = src_datamodule
        print(f'[Trainer] Starting training on source dataset: {src_dataset_name}')
        approach.fit()
        approach.validate()
        
        checkpoint_path = self.args.ckpt_path or approach.checkpoint_path
        
        self.dm.update_log_dir(trg_dataset_name) # Switch the log_dir to trg
        
        # Adaptation to trg
        approach = self._reset_approach(checkpoint_path)
        approach.datamodule = trg_datamodule
        print(f'[Trainer] Starting training on target dataset: {trg_dataset_name}')
        approach.adapt()
        approach.validate()
        
        # Test on trg
        print(f'[Trainer] Starting test on target dataset: {trg_dataset_name}')
        approach.datamodule.set_test_dataset(trg_splits.test)
        approach.test()
        
        self.dm.update_log_dir(src_dataset_name) # Switch the log_dir to src
        
        # Test on src
        print(f'[Trainer] Starting test on source dataset: {src_dataset_name}')
        approach.datamodule.set_test_dataset(src_splits.test)
        approach.test()
        
        
    def _reset_approach(self, checkpoint_path):        
        approach = get_approach(approach_name=self.args.approach, **self.dict_args)
        # Loads checkpoint from the first task or from checkpoint_path
        approach.load_checkpoint(checkpoint_path) 
        print(f'[Trainer] Loaded approach state from {checkpoint_path}')
        return approach    
            
            