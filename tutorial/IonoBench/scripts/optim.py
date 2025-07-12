import torch.nn as nn
import torch.optim as optim

def build_criterion(name: str):
    return {"MSELoss": nn.MSELoss,
            "L1Loss":  nn.L1Loss}.get(name, nn.MSELoss)()

def build_optimizer(name: str, params, lr: float):
    return {"Adam": optim.Adam,
            "SGD":  lambda p, lr: optim.SGD(p, lr, momentum=0.9)}.get(name, optim.Adam)(params, lr=lr)

def build_scheduler(name: str, optim_obj, cfg):
    if name == "ReduceLROnPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optim_obj, factor=0.5, patience=5, verbose=True)
    if name == "CosineAnnealing":
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optim_obj, T_0=10, T_mult=2, eta_min=cfg.train.lr_end)
    return None
