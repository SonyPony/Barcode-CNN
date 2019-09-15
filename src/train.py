# -*- coding: utf-8 -*-
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import torch
import torch.nn.functional as F
import torch.cuda
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
import numpy as np
from dataloader.mas import MASDataset
from torchvision import transforms, utils
from model.p_net import PNet, RNet
from tensorboardX import SummaryWriter


def pnet_loss(bbox_loss, cls_loss):
    return cls_loss + 0.5 * bbox_loss

# ----------------------------------------------------------
def run_epoch(model, optimizer, data_loader, dataset_size, backward=True):
    epoch_loss = 0.
    epoch_acc = 0.

    for i, sample in enumerate(data_loader):
        image, gt_label, gt_bbox = sample["image"].to(dev), sample["label"].to(dev), sample["bbox"].to(dev)

        pred_bbox, pred_label = model(image)
        # HACK zero loss for bbox in negative data
        pred_bbox = pred_bbox.unsqueeze(dim=2).unsqueeze(dim=2)
        pred_bbox[gt_label == 0] = torch.tensor((0., 0., 0., 0.)).unsqueeze(dim=1).unsqueeze(dim=1).to(dev)
        #pred_bbox[gt_label == 0] = torch.tensor((0., 0., 0., 0.)).to(dev)

        bbox_loss = bbox_criterion(pred_bbox, gt_bbox.float())
        #cls_loss = cls_criterion(pred_label, gt_label.unsqueeze(dim=1).unsqueeze(dim=1).long())
        cls_loss = cls_criterion(pred_label, gt_label.long())
        loss = pnet_loss(bbox_loss, cls_loss)

        epoch_loss += loss.item() * image.shape[0]
        # running_loss += loss.item()

        if backward:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pred_label = F.softmax(pred_label)
        #acc = torch.sum(
        #    torch.argmax(pred_label.squeeze(dim=2).squeeze(dim=2), dim=1) == gt_label.long()).detach().cpu().item()
        acc = torch.sum(
            torch.argmax(pred_label, dim=1) == gt_label.long()).detach().cpu().item()
        # running_acc += acc
        epoch_acc += acc

    return epoch_loss / float(dataset_size), epoch_acc / float(dataset_size)

EPOCHS_COUNT = 500
VIEW_STEP = 100
BATCH_SIZE = 64
LR = 1e-4
EXPERIMENTS_DIR = "../experiment/02"
LOG_DIR = "{}/log".format(EXPERIMENTS_DIR)
TRAIN_DATA_DIR = "../dataset/train_data_rnet"
VAL_DATA_DIR = "../dataset/val_data_rnet"
LOAD_MODEL_PATH = "model/weight/rnet_model_v2.pth"
#LOAD_MODEL_PATH = "{}/model_exp_1_2_3.pth".format(EXPERIMENTS_DIR)
start_epoch = 0
current_epoch = 0

# INIT
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device", dev)

train_dataset = MASDataset(directory=TRAIN_DATA_DIR, transform=transforms.ToTensor())
val_dataset = MASDataset(directory=VAL_DATA_DIR, transform=transforms.ToTensor())

train_dataset_len = len(train_dataset)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = RNet()
data = torch.load(LOAD_MODEL_PATH)
if data.keys() == {"weights", "epoch"}:
    start_epoch = data["epoch"]
    data = data["weights"]

#model.load_state_dict(data)
model = model.to(dev)
optimizer = torch.optim.SGD(params=model.parameters(), lr=LR, weight_decay=0)
cls_criterion = nn.CrossEntropyLoss()
bbox_criterion = nn.MSELoss()

# TRAINING
try:
    print("Start training...")
    for e in range(1, EPOCHS_COUNT + 1):
        model.train()
        train_loss, train_acc = run_epoch(model, optimizer, train_loader, train_dataset_len, backward=True)

        model.eval()
        val_loss, val_acc = run_epoch(model, optimizer, val_loader, len(val_dataset), backward=False)

        current_epoch = e
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
    weights = model.state_dict()
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


    torch.save({"weights": weights, "epoch": epoch}, out_path)