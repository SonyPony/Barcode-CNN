# -*- coding: utf-8 -*-
import inspect
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import argparse
import torch
import torch.nn.functional as F
import torch.cuda
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
import numpy as np
from dataloader.mas import MASDataset
from torchvision import transforms, utils
import model as zoo
from torch.utils.tensorboard import SummaryWriter
import model.util as model_util
import wandb


models = dict(inspect.getmembers(zoo, lambda x : inspect.isclass(x) and issubclass(x, nn.Module)))

parser = argparse.ArgumentParser()
parser.add_argument("--epoch_count", action="store", required=True, type=int)
parser.add_argument("--train_dir", action="store", required=True, nargs='+')
parser.add_argument("--val_dir", action="store", required=True)
parser.add_argument("--model", action="store")
parser.add_argument("--lr", action="store", type=float, default=1e-4)
parser.add_argument("--batch_size", action="store", type=int, default=64)
parser.add_argument("--grayscale", action="store", type=int, default=0)
parser.add_argument("--model_type", action="store", required=True)
parser.add_argument("--gradient", action="store", type=int, default=0)

args = parser.parse_args()

EPOCHS_COUNT = args.epoch_count
BATCH_SIZE = args.batch_size
LR = args.lr
TRAIN_DATA_DIR = args.train_dir
VAL_DATA_DIR = args.val_dir
LOAD_MODEL_PATH = args.model
GRADIENT = args.gradient
# Login into wandb
os.system("wandb login 3cd8a6fff84595b3101d186ceece689768431faf")

class PredictionSampleHolder:
    def __init__(self):
        self.image, self.gt_bbox, self.pred_bbox, self.gt_label, self.pred_label = (None, ) * 5

    def __call__(self, i, pred_bbox, pred_label, gt_label, gt_bbox, image):
        if i:
            return
        self.image = image
        self.pred_bbox = pred_bbox
        self.pred_label = pred_label
        self.gt_label = gt_label
        self.gt_bbox = gt_bbox

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        i = self._i
        if i >= self.image.size()[0]:
            raise StopIteration
        self._i += 1
        return self.image[i], self.pred_bbox[i], self.pred_label[i], self.gt_bbox[i], self.gt_label[i]

# ----------------------------------------------------------
def run_epoch(model, optimizer, data_loader, dataset_size, backward=True, prediction_callback=None):
    epoch_loss = 0.
    epoch_acc = 0.
    output_flattened = isinstance(model, (zoo.ONetBase, zoo.RNetBase))

    for i, sample in enumerate(data_loader):
        image, gt_label, gt_bbox = sample["image"].to(dev), sample["label"].to(dev), sample["bbox"].to(dev)

        pred_bbox, pred_label = model(image)
        if prediction_callback:
            prediction_callback(i, pred_bbox, pred_label, gt_label, gt_bbox, image)

        if not output_flattened:
            # HACK zero loss for bbox in negative data
            pred_bbox[gt_label == 0] = torch.tensor((0., 0., 0., 0.)).unsqueeze(dim=1).unsqueeze(dim=1).to(dev)
            gt_label = gt_label.unsqueeze(dim=1).unsqueeze(dim=1)
            gt_bbox = gt_bbox.unsqueeze(dim=2).unsqueeze(dim=2)

        else:
            pred_bbox[gt_label == 0] = torch.tensor((0., 0., 0., 0.)).to(dev)

        bbox_loss = bbox_criterion(pred_bbox, gt_bbox.float())
        cls_loss = cls_criterion(pred_label, gt_label.long())
        loss = model.loss_function(bbox_loss, cls_loss)

        epoch_loss += loss.item() * image.shape[0]

        if backward:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pred_label = F.softmax(pred_label, dim=1)
        if not output_flattened:
            pred_label = pred_label.squeeze(dim=1).squeeze(dim=1)

        acc = torch.sum(
            torch.argmax(pred_label, dim=1) == gt_label.long()).detach().cpu().item()
        epoch_acc += acc

    return epoch_loss / float(dataset_size), epoch_acc / float(dataset_size)

start_epoch = 0
current_epoch = 0
best_acc = 0.

# INIT
wandb.init(project="barcode-detection", config=vars(args))
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device", dev)

# loading dataset
train_dataset = MASDataset(directories=TRAIN_DATA_DIR, transform=transforms.ToTensor(), grayscale=args.grayscale, gradient=GRADIENT)
val_dataset = MASDataset(directories=VAL_DATA_DIR, transform=transforms.ToTensor(), grayscale=args.grayscale, gradient=GRADIENT)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# loading pretrained model
model = models[args.model_type]()   # type: nn.Module
if LOAD_MODEL_PATH:
    start_epoch, best_acc = model_util.load(model, data=torch.load(LOAD_MODEL_PATH))

# init model and loss
model = model.to(dev)
optimizer = torch.optim.SGD(params=model.parameters(), lr=LR, weight_decay=0)
cls_criterion = nn.CrossEntropyLoss()
bbox_criterion = nn.MSELoss()
predction_samples_holder = PredictionSampleHolder()

# TRAINING
print("Start training...")
for e in range(1, EPOCHS_COUNT + 1):
    model = model.train()
    train_loss, train_acc = run_epoch(model, optimizer, train_loader, len(train_dataset), backward=True)

    model = model.eval()
    val_loss, val_acc = run_epoch(
        model,
        optimizer,
        val_loader,
        len(val_dataset),
        backward=False,
        prediction_callback=predction_samples_holder
    )

    # saving and printing
    print("Epoch: {} ({}): Train loss: {:.4f} Train acc: {:.4f} "
          "Val Loss: {:.4f} Val acc: {:.4f}"
          .format(start_epoch + e, e, train_loss, train_acc, val_loss, val_acc))

    wandb.log({
        "Training Loss": train_loss,
        "Training Accuracy": train_acc,
        "Validation Loss": val_loss,
        "Validation Accuracy": val_acc,
        "Examples": [
            wandb.Image(img, caption="Pred: {1}({0:.2f}) GT: {2}".format(
                *(F.softmax(pred_label.detach()).squeeze().max(dim=0)), gt_label))
            for img, _, pred_label, _, gt_label in predction_samples_holder
        ]
    })
# saving the end model
torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))