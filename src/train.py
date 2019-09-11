# -*- coding: utf-8 -*-
import torch
import torch.cuda
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
import numpy as np
from dataloader.mas import MASDataset
from torchvision import transforms, utils
from model.p_net import PNet
from tensorboardX import SummaryWriter


dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device", dev)

train_dataset = MASDataset("../dataset/train_data", transform=transforms.ToTensor())
val_dataset = MASDataset("../dataset/val_data", transform=transforms.ToTensor())

train_dataset_len = len(train_dataset)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

model = PNet()
weights = torch.load("model/weight/pnet_model_v2.pth")
model.load_state_dict(weights)
model = model.to(dev)
optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-2, weight_decay=0)
cls_criterion = nn.CrossEntropyLoss()
bbox_criterion = nn.MSELoss()

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
        pred_bbox[gt_label == 0] = torch.tensor((0., 0., 0., 0.)).unsqueeze(dim=1).unsqueeze(dim=1).to(dev)

        bbox_loss = bbox_criterion(pred_bbox, gt_bbox.float())
        cls_loss = cls_criterion(pred_label, gt_label.unsqueeze(dim=1).unsqueeze(dim=1).long())
        loss = pnet_loss(bbox_loss, cls_loss)

        epoch_loss += loss.item() * image.shape[0]
        # running_loss += loss.item()

        acc = torch.sum(
            torch.argmax(pred_label.squeeze(dim=2).squeeze(dim=2), dim=1) == gt_label.long()).detach().cpu().item()
        # running_acc += acc
        epoch_acc += acc * image.shape[0]

        if backward:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return epoch_loss / float(dataset_size), epoch_acc / float(dataset_size)

EPOCHS_COUNT = 100
VIEW_STEP = 100
BATCH_SIZE = 64
writer = SummaryWriter(log_dir="../experiment/01")

for e in range(EPOCHS_COUNT):
    model.train()
    train_loss, train_acc = run_epoch(model, optimizer, train_loader, train_dataset_len, backward=True)

    model.eval()
    val_loss, val_acc = run_epoch(model, optimizer, val_loader, len(val_dataset), backward=False)
    """epoch_loss = 0.
    epoch_acc = 0.

    for i, sample in enumerate(train_loader):
        image, gt_label, gt_bbox = sample["image"].to(dev), sample["label"].to(dev), sample["bbox"].to(dev)

        pred_bbox, pred_label = model(image)
        # HACK zero loss for bbox in negative data
        pred_bbox[gt_label == 0] = torch.tensor((0., 0., 0., 0.)).unsqueeze(dim=1).unsqueeze(dim=1).to(dev)

        bbox_loss = bbox_criterion(pred_bbox, gt_bbox.float())
        cls_loss = cls_criterion(pred_label, gt_label.unsqueeze(dim=1).unsqueeze(dim=1).long())
        loss = pnet_loss(bbox_loss, cls_loss)

        epoch_loss += loss.item() * image.shape[0]
        #running_loss += loss.item()

        acc = torch.sum(torch.argmax(pred_label.squeeze(dim=2).squeeze(dim=2), dim=1) == gt_label.long()).detach().cpu().item()
        #running_acc += acc
        epoch_acc += acc * image.shape[0]
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % VIEW_STEP == VIEW_STEP - 1:
            print("Loss {:.4f} Acc {:.4f}".format(running_loss / VIEW_STEP, running_acc / VIEW_STEP))
            running_loss = 0.0
            running_acc = 0."""
    print("Epoch: {}, Train loss {:.4f} Train acc {:.4f} Val Loss {:.4f} Val acc {:.4f}"
          .format(e, train_loss, train_acc, val_loss, val_acc))
    writer.add_scalars("data/Loss", {"train": train_loss}, e)
    writer.add_scalars("data/Loss", {"val": val_loss}, e)
    writer.add_scalars("data/Accuracy", {"train": train_acc}, e)
    writer.add_scalars("data/Accuracy", {"val": val_acc}, e)
