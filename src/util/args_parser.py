from argparse import ArgumentParser

from data.data_module import DataModule
from util.config import load_config
from util.directory_manager import DirectoryManager
from approach import (
    RandomForest,
    XGB,
    KNN,
    Baseline,
    NegativeMargin,
    RFS,
    ProtoNet,
    MetaOptNet,
    MAML,
    RelationNet,
    get_approach_type,
    is_approach_transfer_learning,
    is_approach_meta_learning,
)


def parse_arguments():
    cf = load_config()
    
    # Experiment args
    parser = ArgumentParser(conflict_handler='resolve', add_help=True) 
    parser = DataModule.add_argparse_args(parser)
    parser = RandomForest.add_appr_specific_args(parser)
    parser = XGB.add_appr_specific_args(parser)
    parser = KNN.add_appr_specific_args(parser)
    parser = Baseline.add_appr_specific_args(parser)
    parser = NegativeMargin.add_appr_specific_args(parser)
    parser = RFS.add_appr_specific_args(parser)
    parser = ProtoNet.add_appr_specific_args(parser)
    parser = MetaOptNet.add_appr_specific_args(parser)
    parser = MAML.add_appr_specific_args(parser)
    parser = RelationNet.add_appr_specific_args(parser)
    parser.add_argument('--seed', type=int, default=cf['seed'], help='Seed for reproducibility')
    parser.add_argument('--gpu', action='store_true', default=cf['gpu'], help='Use GPU if available')
    parser.add_argument('--n-thr', type=int, default=cf['n_thr'], help='Number of threads')
    parser.add_argument('--log-dir', type=str, default=cf['log_dir'], help='Log directory')
    parser.add_argument('--approach', type=str, default=cf['approach'], help='ML or DL approach to use')
    parser.add_argument('--network', type=str, default=cf['network'], help='Network to use')
    parser.add_argument('--ckpt-path', type=str, default=cf['ckpt_path'], 
                        help='Path to the .pt file containing the state of an approach')
    parser.add_argument('--skip-t1', action='store_true', default=cf['skip_t1'], 
                        help='Skip the first task on src dataset, used only when n_task 2') # TODO: implement
    parser.add_argument('--skip-t2', action='store_true', default=cf['skip_t2'], 
                        help='Skip the second task on trg dataset, used only when n_task 2') # TODO: implement
    # Data args
    parser.add_argument('--datasets', type=str, default=cf['datasets'], 
                        help='Datasets to use', nargs='+', metavar='DATASET')
    parser.add_argument('--is-flat', action='store_true', default=cf['is_flat'],
                        help='Flat the PSQ input')
    parser.add_argument('--return-quintuple', action='store_true', default=cf['return_quintuple'],
                        help='Return the quintuple as well as the data and labels')
    parser.add_argument('--num-pkts', type=int, default=cf['num_pkts'], 
                        help='Number of packets to consider in each biflow')
    parser.add_argument('--fields', type=str, default=cf['fields'],  
                        choices=['PL', 'IAT', 'DIR', 'WIN', 'FLG', 'TTL'],
                        help='Field or fields used (default=%(default)s)', 
                        nargs='+', metavar='FIELD')
    
    args = parser.parse_args()
    
    args.appr_type = get_approach_type(args.approach) 
    args.is_appr_tl = is_approach_transfer_learning(args.approach)
    args.is_appr_meta = is_approach_meta_learning(args.approach)
    args.is_fsl = args.is_appr_tl or args.is_appr_meta

    if len(args.datasets) == 0:
        raise ValueError('At least one dataset must be specified.')
    
    if args.is_fsl and len(args.datasets) != 2:
        raise ValueError(
            'FSL approaches require exactly 2 datasets '
            '(non-few source + few target).'
        )
        
    if args.skip_t1 and args.is_appr_tl:
        print('WARNING: skipping training on src dataset')
        
    if args.skip_t2 and args.is_appr_tl:
        print('WARNING: skipping adaptation on trg dataset')
        
    # Create log dir
    DirectoryManager(args.log_dir)
    
    return args