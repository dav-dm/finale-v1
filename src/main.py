from data.data_manager import DataManager
from util.logger import Logger

from util.args_parser import parse_arguments
from util.config import config_threads
from util.seed import seed_everything
from trainer.trainer_factory import get_trainer


def main():
    # 1. Parse arguments
    args = parse_arguments()
    
    # 2. Set the seed and number of threads
    seed_everything(args.seed)
    config_threads(args.n_thr)
    
    # 3. Init the DataManager and get the data
    data_manager = DataManager(args)
    datasets = data_manager.prepare_datasets()
    
    # 4. Initialize the Trainer and start the run
    trainer = get_trainer(args, datasets)
    trainer.run()
    
    # 5. Final logging (metrics, graphs, etc.)
    logger = Logger(dataset_names=args.datasets)
    logger.process_results()
    logger.plot_per_epoch_metrics()


if __name__ == '__main__':
    main()
