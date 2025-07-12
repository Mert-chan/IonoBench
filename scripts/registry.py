"""
registry.py
@ Mert-chan 
@ 16 Feb 2025
Registry for model classes in the IonoBench project.
- Automatically imports all model files in the source/models directory.
- Provides a decorator to register models.
- Allows building models from a configuration object.
- Supports device specification for model instantiation.
"""

from __future__ import annotations
from typing import Dict, Type
import importlib, pkgutil, pathlib
import torch.nn as nn

# 1. Registry dictionary and decorator
#=======================================================================
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}

def register_model(name: str):
    """Decorator: @register_model("SimVPv2")"""
    key = name.lower()

    def decorator(cls: Type[nn.Module]):
        if key in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' already registered")
        MODEL_REGISTRY[key] = cls
        return cls

    return decorator
#=======================================================================

# 2. Auto-import every file in source/models/
#=======================================================================
def auto_import_models(base_path: pathlib.Path):
    """
    Import every module inside ``source/models`` so that the
    @register_model decorators execute and fill MODEL_REGISTRY.
    """
    models_pkg = (base_path / "source" / "models").resolve()       # => .../source/models
    pkg_root   = models_pkg.parent                                 # => .../source
    pkg_name   = "source.models"                                   # 
    # Make sure package root is importable
    if str(pkg_root) not in importlib.sys.meta_path:
        import sys
        sys.path.insert(0, str(pkg_root))

    # Walk through each .py file
    for module_info in pkgutil.walk_packages([str(models_pkg)]):
        importlib.import_module(f"{pkg_name}.{module_info.name}")
#========================================================================


# 3. Public helper to build the network
#=======================================================================
def build_model(cfg, base_path: str | pathlib.Path, device: str = "cpu"):
    """
    cfg: Configuration object or dictionary containing model parameters.
    base_path: Path to the project root directory, where the source/models directory is located.
    device: Device to which the model should be moved (default is "cpu").
    Returns:
        An instance of the model class specified in the configuration.
    """
    base_path = pathlib.Path(base_path).resolve()
    auto_import_models(base_path)          # populate registry

    name = cfg.model.name.lower() if hasattr(cfg, "model") else cfg["model"]["name"].lower()

    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'.\nRegistered models: {list(MODEL_REGISTRY.keys())}"
        )

    ModelCls = MODEL_REGISTRY[name]
    model_cfg = cfg.model if hasattr(cfg, "model") else cfg["model"]
    model = ModelCls(model_cfg).to(device)
    return model
#=======================================================================