# -*- coding: utf-8 -*-
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
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import model.util as model_util


parser = argparse.ArgumentParser()
parser.add_argument("--experiment_dir", action="store", required=True)
parser.add_argument("--epoch_count", action="store", required=True, type=int)
parser.add_argument("--train_dir", action="store", required=True)
parser.add_argument("--val_dir", action="store", required=True)
parser.add_argument("--model", action="store")
parser.add_argument("--lr", action="store", type=float, default=1e-4)
parser.add_argument("--batch_size", action="store", type=int, default=64)
parser.add_argument("--grayscale", action="store", type=bool, default=False)

args = parser.parse_args()

EPOCHS_COUNT = args.epoch_count
BATCH_SIZE = args.batch_size
LR = args.lr
EXPERIMENTS_DIR = args.experiment_dir
LOG_DIR = "{}/log".format(EXPERIMENTS_DIR)
TRAIN_DATA_DIR = args.train_dir
VAL_DATA_DIR = args.val_dir
LOAD_MODEL_PATH = args.model

"""
EPOCHS_COUNT = 500
BATCH_SIZE = 32
LR = 1e-4
EXPERIMENTS_DIR = "../experiment/29_pnet_24x24_v2_bn"
LOG_DIR = "{}/log".format(EXPERIMENTS_DIR)
TRAIN_DATA_DIR = "../dataset/syn_rnet_train_data_col_bg"
VAL_DATA_DIR = "../dataset/rnet_val_data_rot_15"
LOAD_MODEL_PATH = "model/weight/pnet_model_v2.pth"
"""

# ----------------------------------------------------------
def run_epoch(model, optimizer, data_loader, dataset_size, backward=True):
    epoch_loss = 0.
    epoch_acc = 0.
    output_flattened = isinstance(model, (zoo.ONetBase, zoo.RNetBase))

    for i, sample in enumerate(data_loader):
        image, gt_label, gt_bbox = sample["image"].to(dev), sample["label"].to(dev), sample["bbox"].to(dev)

        pred_bbox, pred_label = model(image)

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
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device", dev)

train_dataset = MASDataset(directory=TRAIN_DATA_DIR, transform=transforms.ToTensor(), grayscale=args.grayscale)
val_dataset = MASDataset(directory=VAL_DATA_DIR, transform=transforms.ToTensor(), grayscale=args.grayscale)

train_dataset_len = len(train_dataset)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = zoo.ExtPnetA3()   # type: nn.Module
if LOAD_MODEL_PATH:
    start_epoch, best_acc = model_util.load(model, data=torch.load(LOAD_MODEL_PATH))

model = model.to(dev)
optimizer = torch.optim.SGD(params=model.parameters(), lr=LR, weight_decay=0)
cls_criterion = nn.CrossEntropyLoss()
bbox_criterion = nn.MSELoss()

# TRAINING
try:
    print("Start training...")
    for e in range(1, EPOCHS_COUNT + 1):
        model = model.train()
        train_loss, train_acc = run_epoch(model, optimizer, train_loader, train_dataset_len, backward=True)

        model = model.eval()
        val_loss, val_acc = run_epoch(model, optimizer, val_loader, len(val_dataset), backward=False)

        # saving and printing
        current_epoch = e
        if val_acc > best_acc:
            best_acc = val_acc
            out_path = "{dir}/best_model.pth".format(dir=EXPERIMENTS_DIR)
            model_util.save(model, epoch=current_epoch + start_epoch, best_acc=best_acc, out_path=out_path)

        print("Epoch: {} ({}): Train loss: {:.4f} Train acc: {:.4f} "
              "Val Loss: {:.4f} Val acc: {:.4f}"
              .format(start_epoch + e, e, train_loss, train_acc, val_loss, val_acc))

        writer = SummaryWriter(log_dir=LOG_DIR)
        writer.add_scalars("Watching/Loss", {"train": train_loss}, e + start_epoch)
        writer.add_scalars("Watching/Loss", {"val": val_loss}, e + start_epoch)
        writer.add_scalars("Watching/Accuracy", {"train": train_acc}, e + start_epoch)
        writer.add_scalars("Watching/Accuracy", {"val": val_acc}, e + start_epoch)
        writer.close()

# SAVING MODEL
except KeyboardInterrupt:
    epoch = current_epoch + start_epoch

    start_model = LOAD_MODEL_PATH if "model_exp" in LOAD_MODEL_PATH else None
    models = tuple(filter(lambda x: "model_exp" in x, os.listdir(EXPERIMENTS_DIR)))
    current_index = 1

    if len(models):
        current_index = max(map(lambda x: int(x
                                .split("/")[-1]
                                .replace(".pth", "")
                                .split("_")[-1]), models)) + 1
    out_path = "{dir}/model_exp_{index}.pth".format(dir=EXPERIMENTS_DIR, index=current_index)
    if start_model:
        out_path = "{dir}/{basename}_{index}.pth"\
            .format(dir=EXPERIMENTS_DIR, basename=start_model.split("/")[-1].replace(".pth", ""), index=current_index)

    model_util.save(model, epoch=epoch, best_acc=best_acc, out_path=out_path)