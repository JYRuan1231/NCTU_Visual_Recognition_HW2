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
from data_augment.util import *
from data_augment.autoaugment import SVHNPolicy



class SVHNDataset(Dataset):
    def __init__(self, data_folder, is_train, split):
        self.data_folder = data_folder
        self.is_train = is_train

        self.annot_dict = {}
        with open(cfg.annotation_file, 'rb') as file:
            self.annot_dict = pickle.load(file)

        # Random generate training and validation dataset
        image_list = glob.glob(os.path.join(data_folder, "*.png"))
        if split:
            if is_train:
                self.images = sorted(image_list)[:30000]
            else:
                self.images = sorted(image_list)[30000:]

    def __getitem__(self, i):

        data_transforms = {
            'train': transforms.Compose([
                # autoaugent
                SVHNPolicy(),
                transforms.ToTensor(),
            ]),
            'val': transforms.Compose([
                # autoaugent
                transforms.ToTensor(),
            ])
        }

        phase = 'val'
        if self.is_train == True:
            phase = 'train'

        # Read image
        image = Image.open(self.images[i])
        image = image.convert('RGB')
        img_name = os.path.basename(self.images[i])

        # Read objects in this image (bounding boxes, labels)

        (labels, boxes) = self.annot_dict[img_name]
        boxes = torch.FloatTensor(boxes)  # (n_objects, 4)
        labels = torch.LongTensor(labels)  # (n_objects)

        # Apply transformations
        image = data_transforms[phase](image)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["img_name"] = img_name

        return image, target

    def __len__(self):
        return len(self.images)

def collate_fn(batch):
    return tuple(zip(*batch))
