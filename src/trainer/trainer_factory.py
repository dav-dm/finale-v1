from trainer.single_task import SingleTaskTrainer
from trainer.few_shot_learning import FewShotLearningTrainer


def get_trainer(args, datasets):
    """
    Factory function to get the appropriate trainer based on the arguments and datasets provided.
    """
    if args.is_fsl:
        return FewShotLearningTrainer(args, datasets)
    # One-task / independent training
    return SingleTaskTrainer(args, datasets)
