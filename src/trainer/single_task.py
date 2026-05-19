from approach.approach_factory import get_approach
from util.seed import seed_everything
from trainer.base_trainer import BaseTrainer


class SingleTaskTrainer(BaseTrainer):
    """
    Trainer for single-task learning.
    For each dataset, it trains, validates, and tests a specific approach.
    """
    def __init__(self, args, datasets):
        super().__init__(args, datasets)
        self.checkpoints = []
        
        
    def run(self):
        for dataset_name, (splits, datamodule) in self.datasets.items():
            seed_everything(self.args.seed) # Reset seed for each dataset to ensure same initialization
            
            self.dict_args['num_classes'] = splits.num_classes
            self.dm.update_log_dir(dataset_name)
            self._save_dict_args()
            
            approach = get_approach(
                approach_name=self.args.approach, datamodule=datamodule, **self.dict_args
            )

            # Load pretrained checkpoint if provided
            if self.args.ckpt_path:
                approach.load_checkpoint(self.args.ckpt_path)
                print(
                    f'[Trainer] Loaded pretrained checkpoint from '
                    f'{self.args.ckpt_path} for {dataset_name}'
                )

            print('='*100)
            
            # Train, validate, and test
            print(f'[Trainer] Starting training on {dataset_name}')
            approach.fit()
            approach.validate()
            approach.test()
            
            if hasattr(approach, 'checkpoint_path'):
                self.checkpoints.append(approach.checkpoint_path)
  