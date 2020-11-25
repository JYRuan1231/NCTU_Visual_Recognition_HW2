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
from model import backboneNet_efficient
from dataset import SVHNDataset, collate_fn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops.misc import FrozenBatchNorm2d


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def plot_img_bbox(img_path, bbox):
    bbox_back = bbox.astype('float')
    image = cv2.imread(test_path + file)[:, :, ::-1]
    plotted_img = draw_rect(image, bbox_back)
    plt.title(label)
    plt.imshow(plotted_img)
    plt.show()


def process_bbox_iou(bbox, label, score, threshold, iou_threshold):
    for i in range(len(bbox)):
        for j in range(i + 1, len(bbox)):
            b1 = bbox[i]
            b2 = bbox[j]
            if get_iou(b1, b2) >= iou_threshold:
                if score[i] < score[j]:
                    score[i] = 0
                else:
                    score[j] = 0
    bbox = bbox[score > threshold].astype('int')
    label = label[score > threshold]
    score = score[score > threshold]

    return bbox, label, score


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transforms = transforms.Compose([transforms.ToTensor(), ])
    output_json = []
    allFileList = os.listdir(test_path)
    allFileList.sort()
    allFileList.sort(key=lambda x: int(x[:-4]))


    # Load model
    backbone = backboneNet_efficient()  # use efficientnet as our backbone
    backboneFPN = backboneWithFPN(backbone)  # add FPN

    anchor_generator = AnchorGenerator(cfg.anchor_sizes, cfg.aspect_ratios)

    model_ft = FasterRCNN(backboneFPN, num_classes=cfg.num_classes, rpn_anchor_generator=anchor_generator,
                          min_size=cfg.min_size, max_size=cfg.max_size)

    model_ft.load_state_dict(torch.load(cfg.model_name).state_dict())
    model_ft.to(device)


    with open(cfg.json_name, 'w', encoding='utf-8') as json_f:
        for file in allFileList:
            if os.path.isfile(cfg.test_path + file):
                print(file)
                output_dict = {}
                path = test_path + file
                img = Image.open(path).convert('RGB')
                img = data_transforms(img)
                img = img.unsqueeze(0)
                with torch.no_grad():
                    model_ft.eval()
                    img = img.to(device)
                    output = model_ft(img)

                    bbox = output[0]["boxes"].cpu().numpy()
                    label = output[0]["labels"].cpu().numpy()
                    score = output[0]["scores"].cpu().numpy()
                    bbox = bbox[score > score_threshold].astype('int')
                    label = label[score > score_threshold]
                    score = score[score > score_threshold]

                    # remove redundant bounding box
                    bbox, label, score = process_bbox_iou(bbox, label, score, cfg.score_threshold, cfg.IoU_threshold)

                    if plot_img == cfg.plot_img:   # plot image
                         plot_img_bbox(path,bbox)
                    for i in range(bbox.shape[0]):
                        bbox[i] = [bbox[i][1], bbox[i][0], bbox[i][3], bbox[i][2]]

                    output_dict["bbox"] = bbox.tolist()
                    output_dict["label"] = label.tolist()
                    output_dict["score"] = score.tolist()
            output_json.append(output_dict)
        json.dump(output_json, json_f)