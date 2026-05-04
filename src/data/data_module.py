import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader, TensorDataset

from data.batch_sampler import EpisodicBatchSampler, MetaEpisodicBatchSampler, make_meta_collate
from util.config import load_config


class DataModule:
    def __init__(self, train_dataset, val_dataset, test_dataset, **kwargs):
        cf = load_config()

        # General data arguments
        self.batch_size = kwargs.get('batch_size', cf['batch_size'])
        self.adapt_batch_size = kwargs.get('adapt_batch_size', cf['adapt_batch_size'])
        self.num_workers = kwargs.get('num_workers', cf['num_workers'])
        self.pin_memory = kwargs.get('pin_memory', cf['pin_memory'])

        is_fsl =  kwargs.get('is_fsl', False)
        self.approach_type = kwargs.get('appr_type', None)
        self.seed = kwargs.get('seed', cf['seed'])

        # FSL-specific arguments
        self.num_episodes = kwargs.get('num_episodes', cf['num_episodes'])
        self.k = kwargs.get('k', cf['k'])
        self.train_k = kwargs.get('train_k', cf['train_k'])
        self.train_q = kwargs.get('train_q', cf['train_q'])
        self.num_ways = kwargs.get('num_ways', cf['num_ways'])

        # Partitions
        self.train_x, self.train_y = train_dataset
        self.val_x, self.val_y = val_dataset
        self.test_x, self.test_y = test_dataset

    @staticmethod
    def add_argparse_args(parent_parser):
        cf = load_config()
        parser = ArgumentParser(
            parents=[parent_parser],
            add_help=True,
            conflict_handler='resolve',
        )
        parser.add_argument('--batch-size', type=int, default=cf['batch_size'])
        parser.add_argument('--adapt-batch-size', type=int, default=cf['adapt_batch_size'])
        parser.add_argument('--num-workers', type=int, default=cf['num_workers'])
        parser.add_argument('--pin-memory', action='store_true', default=cf['pin_memory'])
        parser.add_argument('--num-episodes', type=int, default=cf['num_episodes'])
        parser.add_argument('--k', type=int, default=cf['k'])
        parser.add_argument('--train-k', type=int, default=cf['train_k'])
        parser.add_argument('--train-q', type=int, default=cf['train_q'])
        parser.add_argument('--num-ways', type=int, default=cf['num_ways'])
        return parser

    #-----------------
    # SETTER FUNCTIONS
    #-----------------
    
    def set_train_dataset(self, dataset):
        self.train_x, self.train_y = dataset

    def set_val_dataset(self, dataset):
        self.val_x, self.val_y = dataset

    def set_test_dataset(self, dataset):
        self.test_x, self.test_y = dataset

    #-----------------
    # STANDARD LOADERS
    #-----------------
    # Non-FSL + source training for transfer learning approaches

    def get_train_data(self):
        if self.approach_type == 'dl':
            return self._make_loader(self.train_x, self.train_y,
                                     self.batch_size, shuffle=True)
        return self.train_x, self.train_y

    def get_val_data(self):
        if self.approach_type == 'dl':
            return self._make_loader(self.val_x, self.val_y,
                                     self.batch_size, shuffle=False)
        return self.val_x, self.val_y

    def get_test_data(self):
        if self.approach_type == 'dl':
            return self._make_loader(self.test_x, self.test_y,
                                     self.batch_size, shuffle=False)
        return self.test_x, self.test_y

    # -----------
    # FSL LOADERS
    # -----------

    def get_episode_data(self, episode):
        """
        Returns a DataLoader over the episode's k*n samples,
        yielding batches of adapt_batch_size.
        """
        sampler = EpisodicBatchSampler(
            labels=self.train_y,
            k=self.k,
            seed=self.seed + episode,
        )
        episode_indices = next(iter(sampler))
        dataset = TensorDataset(
            torch.from_numpy(self.train_x[episode_indices]).float(),
            torch.from_numpy(self.train_y[episode_indices]).float(),
        )
        return DataLoader(
            dataset,
            batch_size=self.adapt_batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed + episode),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def get_meta_episode_data(self, partition):
        """
        Meta-learning episodes for meta-training, -validation, and -testing on source.
        """
        x, y = self._partition_arrays(partition)

        sampler = MetaEpisodicBatchSampler(
            labels=y,
            num_episodes=self.num_episodes,
            num_ways=self.num_ways,
            k_shot=self.train_k,
            q_query=self.train_q,
            seed=self.seed,
        )
        dataset = TensorDataset(
            torch.from_numpy(x).float(),
            torch.from_numpy(y).float(),
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=make_meta_collate(self.num_ways, self.train_k),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    # -----------------
    # UTILITY FUNCTIONS
    # -----------------

    def _make_loader(self, x, y, batch_size, shuffle):
        data_tensor = torch.from_numpy(x).float()
        labels_tensor = torch.from_numpy(y).float()
        return DataLoader(
            TensorDataset(data_tensor, labels_tensor),
            batch_size=batch_size,
            shuffle=shuffle,
            generator=torch.Generator().manual_seed(self.seed),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def _partition_arrays(self, partition):
        if partition == 'train':
            return self.train_x, self.train_y
        if partition == 'val':
            return self.val_x, self.val_y
        if partition == 'test':
            return self.test_x, self.test_y
        raise ValueError(f"Unknown partition '{partition}'. Use 'train', 'val', or 'test'.")