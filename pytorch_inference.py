from __future__ import print_function, division
import os
import numpy as np
from PIL import Image
# import glob
from glob import glob
#import SimpleITK as sitk
from torch import optim
import torch.utils.data
import torch
import torch.nn.functional as F
from collections import OrderedDict

import torch.nn
import torchvision
import matplotlib.pyplot as plt
import natsort
from torch.utils.data.sampler import SubsetRandomSampler
from Data_Loader import Images_Dataset, Images_Dataset_folder, UNextDataset
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split
from PIL import Image

from albumentations.augmentations import transforms
from albumentations.augmentations import geometric
from albumentations import RandomRotate90, Resize
from albumentations.core.composition import Compose, OneOf

import shutil
import random
from Models import Unet_dict, NestedUNet, U_Net, R2U_Net, AttU_Net, R2AttU_Net, UNet
from losses import calc_loss, dice_loss, threshold_predictions_v,threshold_predictions_p
from ploting import plot_kernels, LayerActivations, input_images, plot_grad_flow
from Metrics import dice_coeff, accuracy_score
import time
import argparse
import datetime


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2* iou) / (iou+1)

    try:
        hd95_ = hd95(output_, target_)
    except:
        hd95_ = 0
    
    return iou, dice, hd95_


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_id', default=0, type=int)
    parser.add_argument('--dataseed', default=2981, type=int)
    parser.add_argument('--input_size', default=256, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--dataset', default='busi', type=str)
    parser.add_argument('--log_dir', type=str)
    
    config = parser.parse_args()

    return config


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


config = parse_args()
random_seed = 1029
seed_torch(random_seed)
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU')
else:
    print('CUDA is available. Training on GPU')

device = torch.device("cuda:0" if train_on_gpu else "cpu")

#######################################################
#Setting the basic paramters of the model
#######################################################

shuffle = True
valid_loss_min = np.Inf
batch_size = config.batch_size
num_workers = batch_size
n_iter = 1

pin_memory = False
# if train_on_gpu:
#     pin_memory = True

#######################################################
#Setting up the model
#######################################################

model_id = config.model_id
model_Inputs = [U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet, UNet]
model_names = ['U_Net', 'R2U_Net', 'AttU_Net', 'R2AttU_Net', 'U_Net++', 'New_UNet']
model_name = model_names[model_id]
print('Using model {}'.format(model_name))

def model_unet(model_input, in_channel=3, out_channel=1):
    model_test = model_input(in_channel, out_channel)
    return model_test

model_test = model_unet(model_Inputs[model_id], 3, 1)
model_test.to(device)

#######################################################
#Passing the Dataset of Images and Labels
#######################################################

data_dir = 'data'
dataset_name = config.dataset
print('Using {} dataset'.format(dataset_name))

img_ext = '.png'
mask_ext = '.png'

if dataset_name in ['chase', 'kvasir']:
    img_ext = '.jpg'

if dataset_name == 'busi':
    mask_ext = '_mask.png'
elif dataset_name in ['glas', 'cvc']:
    mask_ext = '.png'
elif dataset_name == 'chase':
    mask_ext = '_1stHO.png'
elif dataset_name == 'kvasir':
    mask_ext = '.jpg'

dataseed = config.dataseed
print('dataseed = ' + str(dataseed))
input_h = config.input_size
input_w = config.input_size
print('input_size = ' + str(config.input_size))
num_classes = 1

img_ids = sorted(
    glob(os.path.join(data_dir, dataset_name, 'images', '*' + img_ext))
)
img_ids.sort()
img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=dataseed)

train_transform = Compose([
    RandomRotate90(),
    # transforms.Flip(),
    geometric.transforms.Flip(),
    Resize(input_h, input_w),
    transforms.Normalize(),
])

val_transform = Compose([
    Resize(input_h, input_w),
    transforms.Normalize(),
])

train_dataset = UNextDataset(
    img_ids=train_img_ids,
    img_dir=os.path.join(data_dir, dataset_name, 'images'),
    mask_dir=os.path.join(data_dir, dataset_name, 'masks'),
    img_ext=img_ext,
    mask_ext=mask_ext,
    num_classes=num_classes,
    transform=train_transform)
val_dataset = UNextDataset(
    img_ids=val_img_ids,
    img_dir=os.path.join(data_dir ,dataset_name, 'images'),
    mask_dir=os.path.join(data_dir, dataset_name, 'masks'),
    img_ext=img_ext,
    mask_ext=mask_ext,
    num_classes=num_classes,
    transform=val_transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    drop_last=False)

#######################################################
#Creating a Folder for every data of the program
#######################################################

file_list = os.listdir(config.log_dir)
for file in file_list:
    model_dir = os.path.join(config.log_dir, file)
    if os.path.isdir(model_dir):
        break

file_list = os.listdir(model_dir)
for file in file_list:
    model_path = os.path.join(model_dir, file)
    if 'pth' in file:
        break

ckpt = torch.load(model_path)
new_state_dict = OrderedDict()
for k, v in ckpt.items():
    if not 'total_ops' in k and not 'total_params' in k:  
        new_state_dict[k] = v
model_test.load_state_dict(new_state_dict)
model_test.eval()

iou_avg_meter = AverageMeter()
dice_avg_meter = AverageMeter()
hd95_avg_meter = AverageMeter()

with torch.no_grad():
    for x1, y1, img_ids in val_loader:
        x1, y1 = x1.to(device), y1.to(device)
        y_pred1 = model_test(x1)
        iou, dice, hd95_ = iou_score(y_pred1, y1)
        y_pred1[y_pred1>0.5] = 1
        y_pred1[y_pred1<=0.5] = 0

        for pred, img_id in zip(y_pred1, img_ids):
            pred_np = pred[0].cpu().numpy()
            pred_np = pred_np.astype(np.uint8)
            pred_np = pred_np * 255
            img = Image.fromarray(pred_np, 'L')
            img.save(os.path.join(config.log_dir, '{}.png'.format(img_id)))

        iou_avg_meter.update(iou, x1.size(0))
        dice_avg_meter.update(dice, x1.size(0))
        hd95_avg_meter.update(hd95_, x1.size(0))

print(model_name)
print('IoU: %.4f' % iou_avg_meter.avg)
print('Dice: %.4f' % dice_avg_meter.avg)
print('HD95: %.4f' % hd95_avg_meter.avg)

with torch.no_grad():
    for x1, y1, img_ids in train_loader:
        x1, y1 = x1.to(device), y1.to(device)
        y_pred1 = model_test(x1)
        iou, dice, hd95_ = iou_score(y_pred1, y1)
        y_pred1[y_pred1>0.5] = 1
        y_pred1[y_pred1<=0.5] = 0

        for pred, img_id in zip(y_pred1, img_ids):
            pred_np = pred[0].cpu().numpy()
            pred_np = pred_np.astype(np.uint8)
            pred_np = pred_np * 255
            img = Image.fromarray(pred_np, 'L')
            img.save(os.path.join(config.log_dir, '{}.png'.format(img_id)))

        iou_avg_meter.update(iou, x1.size(0))
        dice_avg_meter.update(dice, x1.size(0))
        hd95_avg_meter.update(hd95_, x1.size(0))

torch.cuda.empty_cache()
