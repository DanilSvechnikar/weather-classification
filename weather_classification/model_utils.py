"""Utils."""

import multiprocessing as mp
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn


def load_weights_lt_model(
    model_path: Path | str,
    replace_key_dict: str = "model.",
) -> OrderedDict:
    """Returns only the weights of the pytorch lightning model."""
    loaded_model = torch.load(model_path, weights_only=True)
    model_weights = loaded_model["state_dict"]
    for key in list(model_weights):
        model_weights[key.replace(replace_key_dict, "")] = model_weights.pop(key)

    return model_weights


def get_n_workers(num_workers: int) -> int:
    """Returns number of workers."""
    max_cpu_count = mp.cpu_count()
    if num_workers < 0:
        num_workers = max_cpu_count

    num_workers = min(max_cpu_count, num_workers)
    return num_workers


def get_params_num(model: nn.Module, with_grad: bool = False) -> int:
    """Return number of parameters."""
    if with_grad:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    return sum(p.numel() for p in model.parameters())
