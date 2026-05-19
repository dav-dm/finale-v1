import json
from abc import ABC, abstractmethod

from util.directory_manager import DirectoryManager
from util.git import get_git_commit_sha


class BaseTrainer(ABC):
    
    def __init__(self, args, datasets):
        self.args = args
        self.dict_args = vars(args)
        self.datasets = datasets
        self.dm = DirectoryManager()
        self.dict_args['git_commit_sha'] = get_git_commit_sha()
        
    @abstractmethod
    def run(self):
        pass
    
    def _save_dict_args(self):
        with open(f'{self.dm.log_dir}/dict_args.json', 'w') as f:
            json.dump(self.dict_args, f)