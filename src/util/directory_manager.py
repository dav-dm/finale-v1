import time
from pathlib import Path


class DirectoryManager:
    _instance = None

    def __new__(cls, log_dir=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, log_dir=None):
        if hasattr(self, 'initialized'):
            return

        self.base_log_dir = Path(log_dir).resolve()
        self.mkdir_new_exp()
        self.initialized = True

    def _ensure_directory(self, path):
        """
        Create the directory if it doesn't exist.
        """
        path.mkdir(parents=True, exist_ok=True)

    def mkdir(self, path):
        full_path = self.log_dir / path
        full_path.mkdir(parents=True, exist_ok=True)
        return full_path
    
    def mkdir_new_exp(self):
        """
        Create a new experiment directory based on the current timestamp.
        """
        self.exp_dir = self.base_log_dir / f"{round(time.time())}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.exp_dir

    def update_log_dir(self, folder_name):
        """
        Set log directory to <exp_dir>/<folder_name>
        """
        self.log_dir = self.exp_dir / folder_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        return self.log_dir
    