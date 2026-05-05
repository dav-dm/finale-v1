from .ml_module import MLModule
from .dl_module import DLModule
from .random_forest import RandomForest
from .scratch import Scratch
from .baseline import Baseline
from .baseline_pp import BaselinePP
from .negative_margin import NegativeMargin
from .rfs import RFS
from .xgb import XGB
from .knn import KNN

from .approach_factory import (
    get_approach, get_approach_type, is_approach_transfer_learning, is_approach_meta_learning
)