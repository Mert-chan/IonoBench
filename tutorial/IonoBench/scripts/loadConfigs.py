"""
data.py
@ Mert-chan 
@ 16 Feb 2025
- Loads Configurations from a YAML file
- Uses OmegaConf for configuration management
"""

from pathlib import Path
from typing import Union, Dict
from omegaconf import OmegaConf, DictConfig
import os

_CFG_ROOT = Path(__file__).resolve().parent / "configs"     # adapt if needed


def _load_yaml(path: Path) -> DictConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return OmegaConf.load(path)


def load_configs(model: str = "SimVPv2",
                 mode:  str = "train",
                 split: str = "stratified",         # or "chronological"
                 base_path: Path | str = None,
                 cli_overrides: Dict[str, Union[str, int, float, bool]] | None = None,
                 as_dict: bool = False) -> Union[DictConfig, Dict]:
    
    cfg_dir = os.path.join(base_path, "configs")
    cfg_dir = Path(cfg_dir)

    base_cfg   = _load_yaml(cfg_dir / "base.yaml")
    model_cfg  = _load_yaml(cfg_dir / "models" / f"{model}.yaml")
    mode_cfg   = _load_yaml(cfg_dir / "modes"  / f"{mode}.yaml")

    merged = OmegaConf.merge(base_cfg, model_cfg, mode_cfg)
    merged.paths.base_dir = base_path
    merged.data.data_split = split  
    # Apply CLI / programmatic overrides (optional)
    if cli_overrides:
        merged = OmegaConf.merge(merged, OmegaConf.create(cli_overrides))

    # Automatically calculate total_T if it exists in model config and is null
    try:
        if 'model' in merged and 'total_T' in merged.model and merged.model.total_T is None:
            seq_len = merged.get('data', {}).get('seq_len')
            pred_horz = merged.get('data', {}).get('pred_horz')
            if seq_len is not None and pred_horz is not None:
                merged.model.total_T = int(seq_len) + int(pred_horz)
    except Exception as e:
        print(f"Warning: Could not auto-calculate total_T: {e}")

    return OmegaConf.to_container(merged, resolve=True) if as_dict else merged