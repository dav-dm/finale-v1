from trainer.single_task import SingleTaskTrainer
from trainer.transfer_learning import TransferLearningTrainer


def get_trainer(args, datasets):
    """
    Factory function to get the appropriate trainer based on the arguments and datasets provided.
    """
    # FSL paradigms
    if args.is_appr_tl:
        return TransferLearningTrainer(args, datasets)

    # One-task / independent training
    return SingleTaskTrainer(args, datasets)
