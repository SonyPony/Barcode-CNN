# coding=utf-8
import wandb
import os
import torch


class Logger:
    def __init__(self):
        self._best_acc = 0

    def log(self, train_loss, train_acc, val_loss, val_acc, model, **additional):
        wandb.log({
            **{
                "Training Loss": train_loss,
                "Training Accuracy": train_acc,
                "Validation Loss": val_loss,
                "Validation Accuracy": val_acc
            }, **additional})

        if val_acc >= self._best_acc:   # saving the best model
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, "best_model.pt"))
            wandb.run.summary["Best Validation Accuracy"] = val_acc
            self._best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))
