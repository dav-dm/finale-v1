import numpy as np
import torch
from torch.utils.data import Sampler
from functools import partial


class EpisodicBatchSampler(Sampler):
    """
    Sampler that yields a single episode batch.
    Each episode contains k samples per class drawn from the dataset,
    for all classes present in the partition.
    """
    def __init__(self, labels, k, seed):
        super().__init__(None)
        if k < 1:
            raise ValueError(f'k must be at least 1, got {k}')
        labels = np.asarray(labels)
        self.class_indices = {
            int(c): np.where(labels == c)[0] for c in np.unique(labels)
        }
        self.k = k
        self.seed = seed
        small_classes = [c for c, idx in self.class_indices.items() if len(idx) < k]
        if small_classes:
            print(
                f'WARNING: {len(small_classes)} class(es) have fewer than k={k} '
                f'samples: {small_classes}. Sampling with replacement for those.'
            )

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        indices = []
        for cls_idx in self.class_indices.values():
            replace = len(cls_idx) < self.k
            indices.extend(rng.choice(cls_idx, size=self.k, replace=replace).tolist())
        yield indices

    def __len__(self):
        return 1


class MetaEpisodicBatchSampler(Sampler):
    """
    Episodic sampler for meta-learning. Each episode contains num_ways classes,
    and for each class k support samples and q query samples are drawn.
    """
    def __init__(self, labels, num_episodes, num_ways, k_shot, q_query, seed):
        super().__init__(None)

        if num_ways < 2:
            raise ValueError(f"num_ways must be at least 2, got {num_ways}")
        if k_shot < 1:
            raise ValueError(f"k_shot must be at least 1, got {k_shot}")
        if q_query < 1:
            raise ValueError(f"q_query must be at least 1, got {q_query}")

        labels = np.asarray(labels)
        classes = np.unique(labels)

        if len(classes) < num_ways:
            raise ValueError(
                f"The split has only {len(classes)} classes "
                f"but num_ways={num_ways}"
            )

        self.classes = classes
        self.class_indices = {
            int(c): np.where(labels == c)[0] for c in classes
        }

        self.num_episodes = num_episodes
        self.num_ways = num_ways
        self.k = k_shot
        self.q = q_query
        self.seed = seed

        self._validate_parameters()

    def _validate_parameters(self):
        needed = self.k + self.q

        too_few_for_query = []
        no_support_left = []
        support_replacement_classes = []
        n_samples = []

        for cls, idx in self.class_indices.items():
            n = len(idx)

            if n < self.q:
                too_few_for_query.append(cls)
                n_samples.append(n)
            elif n == self.q:
                no_support_left.append(cls)
                n_samples.append(n)
            elif n < needed:
                support_replacement_classes.append(cls)
                n_samples.append(n)

        if too_few_for_query:
            raise ValueError(
                f"Some classes have fewer than q_query={self.q} samples: "
                f"{too_few_for_query} has {n_samples} samples. Cannot sample query without replacement."
            )
        if no_support_left:
            raise ValueError(
                f"Some classes have exactly q_query={self.q} samples: "
                f"{no_support_left} has {n_samples} samples. No samples would remain for support."
            )
        if support_replacement_classes:
            print(
                f"WARNING: {len(support_replacement_classes)} class(es) have fewer "
                f"than k_shot + q_query samples"
                f": {support_replacement_classes} has {n_samples} samples. "
                f"Sampling support with replacement for those."
            )

    def _sample_class_episode(self, cls, rng):
        idx = self.class_indices[int(cls)]

        query_idx = rng.choice(idx, size=self.q, replace=False) # Sample query without replacement

        query_set = set(query_idx.tolist())
        support_pool = np.array([i for i in idx if i not in query_set]) # Remaining samples for support

        if len(support_pool) >= self.k:
            # If enough samples remain for support, sample without replacement
            support_idx = rng.choice(support_pool, size=self.k, replace=False)
        else:
            # Otherwise, sample with replacement from the support pool
            support_idx = support_pool.tolist()
            missing = self.k - len(support_idx) # Number of additional samples needed for support

            # Sample extra indices with replacement from the support pool to fill the support set
            extra_support_idx = rng.choice(
                support_pool,
                size=missing,
                replace=True,
            )
            # Add the extra sampled indices to the support set
            support_idx.extend(extra_support_idx.tolist())
            support_idx = np.asarray(support_idx)
        # Return support and query indices as lists
        return support_idx.tolist(), query_idx.tolist()

    def __iter__(self):
        for ep in range(self.num_episodes):
            rng = np.random.default_rng(self.seed + ep)

            selected_classes = rng.choice(
                self.classes,
                size=self.num_ways,
                replace=False,
            )

            support_idx = []
            query_idx = []

            for cls in selected_classes:
                cls_support_idx, cls_query_idx = self._sample_class_episode(cls, rng)

                support_idx.extend(cls_support_idx)
                query_idx.extend(cls_query_idx)

            yield support_idx + query_idx

    def __len__(self):
        return self.num_episodes