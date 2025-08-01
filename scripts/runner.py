#####################################################################
"""
runner.py
@ Mert-chan 
13 July 2025 (Last Modified) (In development for future CLI updates)
- Runner class that will be used by the CLI to run experiments
"""
#####################################################################


# Libraries
#====================================================================
import torch
from scripts.registry import MODEL_REGISTRY
from scripts.optim import *
#====================================================================


# Runner class that will be used by the CLI
#====================================================================
class Runner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build model
        self.model = MODEL_REGISTRY[cfg.model.name](cfg).to(self.device)

        # Build criterion/optim/scheduler
        self.criterion = build_criterion(cfg.train.criterion)
        self.optimizer = build_optimizer(cfg.train.optimizer,
                                         self.model.parameters(),
                                         cfg.train.lr)
        self.scheduler = build_scheduler(cfg.train.scheduler,
                                         self.optimizer, cfg)

    def train(self): ...
    def test(self):  ...
#====================================================================