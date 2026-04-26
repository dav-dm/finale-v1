from trainer.single_task import SingleTaskTrainer
from trainer.transfer_learning import TransferLearningTrainer


def get_trainer(args, datasets):
    """
    Factory function to get the appropriate trainer based on the arguments and datasets provided.
    """
    n_datasets = len(datasets)

    if n_datasets == 0:
        raise ValueError('At least one dataset must be specified.')

    # Transfer Learning
    if args.is_appr_tl:
        if n_datasets != 2:
            raise ValueError(
                'Transfer Learning requires exactly 2 datasets '
                '(pretraining + finetuning).'
            )
        return TransferLearningTrainer(args, datasets)

    # One-task / independent training
    return SingleTaskTrainer(args, datasets)
