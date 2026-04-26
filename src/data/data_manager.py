from data.data_module import DataModule
from data.reader import get_data_labels
from data.splits import train_val_test_split


class DataManager:
    """
    Manages and prepares data for training, validation, and testing.
    """
    def __init__(self, args):
        self.args = args
        self.seed = self.args.seed
        
    def prepare_datasets(self):
        dataset_args = {
            'num_pkts': self.args.num_pkts,
            'fields': self.args.fields,
            'is_flat': self.args.is_flat,
            'seed': self.args.seed,
        }
        fsl_args = {
            'is_fsl': self.args.is_fsl,
            'k_seed': self.args.k_seed,
            'test_k': self.args.test_k,
        }
        datasets = dict()
        
        for dataset_name in self.args.datasets:
            dataset = get_data_labels(dataset=dataset_name, **dataset_args)
            splits = train_val_test_split(
                dataset=dataset, seed=self.seed, return_quintuple=self.args.return_quintuple,
                fsl_kwargs=fsl_args if self.args.is_fsl else None
            )
            datamodule = DataModule(
                train_dataset=splits.train[:2], # Do not include quintuples
                val_dataset=splits.val[:2],
                test_dataset=splits.test[:2],
                **vars(self.args)
            )
            
            datasets[dataset_name] = (splits, datamodule)
            
        return datasets
            