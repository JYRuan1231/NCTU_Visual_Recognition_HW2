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
from dataset import SVHNDataset, collate_fn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops.misc import FrozenBatchNorm2d


class backboneWithFPN(nn.Module):
    def __init__(self, backbone):
        super(backboneWithFPN, self).__init__()

        # resnet
        #         self.body = IntermediateLayerGetter(backbone, return_layers={'layer2': '1', 'layer3': '2', 'layer4': '3'})
        # efficientnet
        self.body = IntermediateLayerGetter(backbone,
                                            return_layers={'block3': '0', 'block4': '1', 'block5': '2', 'block6': '3'})
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[112, 160, 272, 448],
            out_channels=112,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = 112

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x


class backboneNet_efficient(nn.Module):
    def __init__(self):
        super(backboneNet_efficient, self).__init__()
        net = timm.create_model('tf_efficientnet_b4', pretrained=True)

        layers_to_train = ['blocks']

        for name, parameter in net.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)

        self.conv_stem = net.conv_stem
        self.bn1 = net.bn1
        self.act1 = net.act1
        self.block0 = net.blocks[0]
        self.block1 = net.blocks[1]
        self.block2 = net.blocks[2]
        self.block3 = net.blocks[3]
        self.block4 = net.blocks[4]
        self.block5 = net.blocks[5]
        self.block6 = net.blocks[6]
        self.conv_head = net.conv_head
        self.bn2 = net.bn2
        self.act2 = net.act2

    def forward(self, x):

        x1 = self.conv_stem(x)
        x2 = self.bn1(x1)
        x3 = self.act1(x2)
        x4 = self.block0(x3)
        x5 = self.block1(x4)
        x6 = self.block2(x5)
        x7 = self.block3(x6)
        x8 = self.block4(x7)
        x9 = self.block5(x8)
        x10 = self.block6(x9)
        x11 = self.conv_head(x10)
        x12 = self.bn2(x11)
        x13 = self.act2(x12)

        return x13