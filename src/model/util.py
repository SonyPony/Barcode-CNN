# coding=utf-8
import torch
import logging
import torch.nn as nn
from typing import Dict


def save(model, epoch, best_acc, out_path):
    torch.save({"weights": model.state_dict(), "epoch": epoch, "best_acc": best_acc}, out_path)

def load(model: nn.Module, data: Dict):
    weights = data
    epoch = 0
    best_acc = 0.

    if "weights" in data.keys():
        weights = data["weights"]

    elif "epoch" in data.keys():
        epoch = data["epoch"]

    elif "best_acc" in data.keys():
        best_acc = data["best_acc"]

    model_orig_weights = model.state_dict()
    loaded_weights = 0

    for k, v in weights.items():
        if k in model_orig_weights.keys():
            loaded_weights += 1
            model_orig_weights[k] = v
    if not loaded_weights:
        logging.log(logging.WARNING, "Model did not loaded weights.")

    model.load_state_dict(model_orig_weights)

    return epoch, best_acc