import torch
import torch.nn as nn
import torchvision
import csv
import timm
import time
import glob
import copy
import os
import json
import cv2
import numpy as np
import pickle
import config as cfg
import torch.optim as optim
from torch.optim import lr_scheduler
from timm.models import *
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils, models, datasets
from PIL import Image
from PIL import Image, ImageEnhance, ImageOps
from data_augment.data_aug import *
from data_augment.bbox_util import *
from data_augment.util import *
from data_augment.autoaugment import SVHNPolicy
from model import backboneNet_efficient, backboneWithFPN
from dataset import SVHNDataset, collate_fn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops.misc import FrozenBatchNorm2d


def train_model(
    model, train_loader, valid_loader, optimizer, scheduler, num_epochs=25
):
    iter_start = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0
    best_mAP = 0.0
    train_size = len(train_loader)
    valid_size = len(valid_loader)

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
                model_loader = train_loader
                dataset_size = train_size
            if phase == "val":
                model.train()  # Set model to validation mode
                model_loader = valid_loader
                dataset_size = valid_size

            running_loss = 0.0

            # Iterate over data.
            if phase == "train":
                for iter, (images, targets) in enumerate(model_loader):
                    images = list(image.to(device) for image in images)
                    targets = [
                        {k: v.to(device) for k, v in t.items()}
                        for t in targets
                    ]
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(images, targets)
                        loss = sum(loss for loss in outputs.values())
                        running_loss += loss
                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()
                    if (iter + 1) % 100 == 0:
                        iter_end = time.time() - iter_start
                        print(
                            "iter:{} Loss: {:.4f} Time: {:.0f}m {:.0f}s".format(
                                iter, loss, iter_end // 60, iter_end % 60
                            )
                        )
                        iter_start = time.time()
                scheduler.step()
                epoch_loss = running_loss / dataset_size
                print("{} Loss: {:.4f}".format(phase, running_loss))

                if best_loss > running_loss:
                    best_loss = running_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print("save best training weight,complete!")
        time_elapsed = time.time() - since
        print(
            "Complete one epoch in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )

    print(
        "Best val Acc: {:4f} Best val mAP: {:4f}".format(best_loss, best_mAP)
    )
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 10:1
    train_dataset = SVHNDataset(cfg.train_path, True, True)
    valid_dataset = SVHNDataset(cfg.train_path, False, True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.workers,
        collate_fn=collate_fn,
    )

    # Start to traning FASTER RCNN
    backbone = backboneNet_efficient()  # use efficientnet as our backbone
    backboneFPN = backboneWithFPN(backbone)  # add FPN

    anchor_generator = AnchorGenerator(cfg.anchor_sizes, cfg.aspect_ratios)

    model_ft = FasterRCNN(
        backboneFPN,
        num_classes=cfg.num_classes,
        rpn_anchor_generator=anchor_generator,
        min_size=cfg.min_size,
        max_size=cfg.max_size,
    )

    model_ft.to(device)

    optimizer_ft = optim.SGD(
        model_ft.parameters(),
        lr=cfg.learning_rate,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    lr_scheduler = lr_scheduler.MultiStepLR(
        optimizer_ft, milestones=cfg.milestones, gamma=cfg.gamma
    )

    model_ft = train_model(
        model_ft,
        train_loader,
        valid_loader,
        optimizer_ft,
        lr_scheduler,
        num_epochs=cfg.epochs,
    )

    torch.save(model_ft.state_dict(), cfg.model_folder + cfg.model_name)
