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
    parser.add_argument('--epoch', default=400, type=int)
    parser.add_argument('--dataset', default='busi', type=str)
    parser.add_argument('--initial_lr', default=1e-4, type=float)
    
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

batch_size = config.batch_size
epoch = config.epoch
shuffle = True
valid_loss_min = np.Inf
num_workers = config.batch_size
lossT = []
lossL = []
lossL.append(np.inf)
lossT.append(np.inf)
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

from pdb import set_trace as st
from thop import profile
import time 

def count_parameters(model_test):
    return sum(p.numel() for p in model_test.parameters() if p.requires_grad)

params = count_parameters(model_test)
params_in_million = params / 1e6
print(f"模型参数量：{params_in_million} 百万")

model_test.eval()
input_tensor = torch.randn(config.batch_size, 3, config.input_size, config.input_size).cuda()
# 将输入传递给模型并测量推理时间
with torch.no_grad():
    start_time = time.time()
    output = model_test(input_tensor)
    end_time = time.time()
# 计算推理时间
inference_time_ms = (end_time - start_time) * 1000
print(f"推理时间：{inference_time_ms} 毫秒")

# 运行模型并测量GFLOPs
with torch.no_grad():
    flops, _ = profile(model_test, inputs=(input_tensor,))
    gflops = flops / 1e9
    print(f"GFLOPs：{gflops} G")

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
#Using Adam as Optimizer
#######################################################

initial_lr = 0.001
opt = torch.optim.Adam(model_test.parameters(), lr=initial_lr) # try SGD
#opt = optim.SGD(model_test.parameters(), lr = initial_lr, momentum=0.99)

MAX_STEP = int(1e10)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_STEP, eta_min=1e-5)
#scheduler = optim.lr_scheduler.CosineAnnealingLr(opt, epoch, 1)

#######################################################
#Creating a Folder for every data of the program
#######################################################

now = datetime.datetime.now()
formatted_time = now.strftime("%m%d%H%M")
New_folder = './output/{}_{}_{}_{}'.format(model_name, dataset_name, dataseed, formatted_time)

if os.path.exists(New_folder) and os.path.isdir(New_folder):
    import ipdb; ipdb.set_trace()

try:
    os.mkdir(New_folder)
except OSError:
    print("Creation of the main directory '%s' failed " % New_folder)
else:
    print("Successfully created the main directory '%s' " % New_folder)

my_writer = SummaryWriter(New_folder)

#######################################################
#checking if the model exists and if true then delete
#######################################################

read_model_path = os.path.join(New_folder, 'Unet_D_' + str(epoch) + '_' + str(batch_size))

if os.path.exists(read_model_path) and os.path.isdir(read_model_path):
    shutil.rmtree(read_model_path)
    print('Model folder there, so deleted for newer one')

try:
    os.mkdir(read_model_path)
except OSError:
    print("Creation of the model directory '%s' failed" % read_model_path)
else:
    print("Successfully created the model directory '%s' " % read_model_path)

#######################################################
#Training loop
#######################################################

for i in range(epoch):
    train_loss = 0.0
    valid_loss = 0.0
    since = time.time()
    scheduler.step(i)
    lr = scheduler.get_lr()

    #######################################################
    #Training Data
    #######################################################

    model_test.train()
    k = 1

    for x, y, img_ids in train_loader:
        x, y = x.to(device), y.to(device)

        #If want to get the input images with their Augmentation - To check the data flowing in net
        # input_images(x, y, i, n_iter, k)

        opt.zero_grad()

        y_pred = model_test(x)
        lossT = calc_loss(y_pred, y)     # Dice_loss Used

        train_loss += lossT.item() * x.size(0)
        lossT.backward()
        opt.step()
        x_size = lossT.item() * x.size(0)
        k = 2

    #######################################################
    #Validation Step
    #######################################################

    model_test.eval()

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    hd95_avg_meter = AverageMeter()
    with torch.no_grad():
        for x1, y1, img_ids in val_loader:
            x1, y1 = x1.to(device), y1.to(device)

            y_pred1 = model_test(x1)
            lossL = calc_loss(y_pred1, y1)
            iou, dice, hd95_ = iou_score(y_pred1, y1)
            iou_avg_meter.update(iou, x1.size(0))
            dice_avg_meter.update(dice, x1.size(0))
            hd95_avg_meter.update(hd95_, x1.size(0))

            valid_loss += lossL.item() * x1.size(0)
            x_size1 = lossL.item() * x1.size(0)
        
        del lossL, y_pred1, x1, y1
        
    train_loss = train_loss / len(train_dataset)
    valid_loss = valid_loss / len(val_dataset)

    my_writer.add_scalar('train/loss', train_loss, global_step=i)
    my_writer.add_scalar('valid/loss', valid_loss, global_step=i)
    my_writer.add_scalar('valid/iou', iou_avg_meter.avg, global_step=i)
    my_writer.add_scalar('valid/dice', dice_avg_meter.avg, global_step=i)

    if (i+1) % 1 == 0:
        print('Epoch: {}/{} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(i + 1, epoch, train_loss, valid_loss))

    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model '.format(valid_loss_min, valid_loss))
        torch.save(model_test.state_dict(), os.path.join(New_folder, 'Unet_D_' + str(epoch) + '_' + str(batch_size), 'Unet_epoch_' + str(epoch) + '_batchsize_' + str(batch_size) + '.pth'))

        valid_loss_min = valid_loss

        print(model_name)
        print('IoU: %.4f' % iou_avg_meter.avg)
        print('Dice: %.4f' % dice_avg_meter.avg)
        print('HD95: %.4f' % hd95_avg_meter.avg)

if torch.cuda.is_available():
    torch.cuda.empty_cache()

model_test.load_state_dict(torch.load(os.path.join(New_folder, 'Unet_D_' + str(epoch) + '_' + str(batch_size), 'Unet_epoch_' + str(epoch) + '_batchsize_' + str(batch_size) + '.pth')))

model_test.eval()

iou_avg_meter = AverageMeter()
dice_avg_meter = AverageMeter()
hd95_avg_meter = AverageMeter()

with torch.no_grad():
    for x1, y1, img_ids in val_loader:
        x1, y1 = x1.to(device), y1.to(device)
        y_pred1 = model_test(x1)
        iou, dice, hd95_ = iou_score(y_pred1, y1)
        y_pred1[y_pred1>0.5] = 255
        y_pred1[y_pred1<=0.5] = 0

        for pred, img_id in zip(y_pred1, img_ids):
            pred_np = pred[0].cpu().numpy()
            pred_np = pred_np.astype(np.uint8)
            pred_np = pred_np * 255
            img = Image.fromarray(pred_np, 'L')
            img.save(os.path.join(New_folder, '{}.png'.format(img_id)))

        iou_avg_meter.update(iou, x1.size(0))
        dice_avg_meter.update(dice, x1.size(0))
        hd95_avg_meter.update(hd95_, x1.size(0))

print(model_name)
print('IoU: %.4f' % iou_avg_meter.avg)
print('Dice: %.4f' % dice_avg_meter.avg)
print('HD95: %.4f' % hd95_avg_meter.avg)

my_writer.add_scalar('test/iou', iou_avg_meter.avg, global_step=i)
my_writer.add_scalar('test/dice', dice_avg_meter.avg, global_step=i)

with torch.no_grad():
    for x1, y1, img_ids in train_loader:
        x1, y1 = x1.to(device), y1.to(device)
        y_pred1 = model_test(x1)
        iou, dice, hd95_ = iou_score(y_pred1, y1)
        y_pred1[y_pred1>0.5] = 255
        y_pred1[y_pred1<=0.5] = 0

        for pred, img_id in zip(y_pred1, img_ids):
            pred_np = pred[0].cpu().numpy()
            pred_np = pred_np.astype(np.uint8)
            pred_np = pred_np * 255
            img = Image.fromarray(pred_np, 'L')
            img.save(os.path.join(New_folder, '{}.png'.format(img_id)))

        iou_avg_meter.update(iou, x1.size(0))
        dice_avg_meter.update(dice, x1.size(0))
        hd95_avg_meter.update(hd95_, x1.size(0))

torch.cuda.empty_cache()
