import torch
import matplotlib.pyplot as plt

import os
import os.path
from os import listdir
from os.path import join
import random
import cv2
import json

import torch.utils.data as data
from PIL import Image

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader

from harmonic import Embedding, AddSine
import argparse

# functions to read data
def default_loader(filepath):
    return Image.open(filepath).convert('RGB')

class Reader(data.Dataset):
    def __init__(self, image_list, labels_list=[], transform=None, target_transform=None, use_cache=True, loader=default_loader):
        
        self.images = image_list
        self.loader = loader
        
        if len(labels_list) is not 0:
            assert len(image_list) == len(labels_list)
            self.labels = labels_list
        else:
            self.labels = False

        self.transform = transform
        self.target_transform = target_transform

        self.cache = {}
        self.use_cache = use_cache

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if idx not in self.cache:           
            img = self.loader(self.images[idx])
            if self.labels:
                target = Image.open(self.labels[idx]).convert('L')
                # target = Image.open(self.labels[idx])
            else:
                target = None
        else:
            img,target = self.cache[idx]
            
        if self.use_cache:
            self.cache[idx] = (img, target)

        seed = np.random.randint(2147483647)
        
        random.seed(seed)
        if self.transform is not None:
            img = self.transform(img)

        random.seed(seed)
        if self.labels:
            if self.target_transform is not None:
                target = self.target_transform(target)
            
        return np.array(img), np.array(target)
    
class ToLogits(object):
    def __init__(self,expand_dim=None):
        self.expand_dim = expand_dim

    def __call__(self, pic):
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int32, copy=True))
        elif pic.mode == 'F':
            img = torch.from_numpy(np.array(pic, np.float32, copy=False))
        elif pic.mode == '1':
            img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if self.expand_dim is not None:
            return img.unsqueeze(self.expand_dim)
        return img
    
def log_weights_norm(gain=1.):
    def f(w):
        w[w < 2] = 2.
        w = gain / torch.log(w)
        return w
    return f

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fit Harmonic Embeddings")
    parser.add_argument('--num_sins', type=int, default=16, help='Number of functions')
    parser.add_argument('--epsilon', type=float, default=2, help='Distance for separation')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--nepoch', type=int, default=700, help='Number of training epochs')
    parser.add_argument('--es_max_epoch', type=int, default=700, help='Early stopping maximum number of epochs')
    parser.add_argument('--op_method', type=str, default='adamw', choices=['adamw', 'sgd'], help='Optimization method')
    parser.add_argument('--bs', type=int, default=64, help='Batch size')
    parser.add_argument('--data_file', type=str, default='../datasets/CVPPP2017/A1/A1_90_coco.json')
    # parser.add_argument('--data_file', type=str, default='../datasets/komatsuna_multi_view/multi_plant/komatsuna_rgb_train_coco_0.json')
    # parser.add_argument('--data_file', type=str, default='../datasets/msu_pid/Release/Dataset/Images/Arabidopsis/msu_pid_arabidopsis_train_coco_0.json')
    parser.add_argument('--img_size', type=int, default=512, help='img size') # 512 for cvppp, 480 for komatsuna, 256 for msu_pid

    args = parser.parse_args()
    return args
args = parse_arguments()

train_json_path = args.data_file
with open(train_json_path,'r') as f:
    train_json = json.load(f)
rgb = []
labels = []
for image in train_json['images']:
    rgb.append(image['file_name'])
    if 'CVPPP' in train_json_path:
        args.img_size = 512
        label_file_path = image['file_name'].replace('rgb','label')
    elif 'komatsuna' in train_json_path:
        args.img_size = 480
        label_file_path = image['file_name'].replace('multi_plant','multi_label').replace('rgb_','label_')
    elif 'msu_pid' in train_json_path:
        args.img_size = 256
        label_file_path = image['file_name'].replace('Images','Labels').replace('_rgb.png','_label_rgb.png')
    labels.append(label_file_path)

if 0 == len(rgb):
    print("No dataset found")
    exit(-1)
assert len(rgb) == len(labels)

# hyperparameters
device = 'cuda'
dimx = args.img_size
dimy = args.img_size
bs = args.bs
transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomVerticalFlip(),
     transforms.RandomResizedCrop((dimx, dimy), scale=(0.7,1.)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])
transform_target = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomVerticalFlip(),
     transforms.RandomResizedCrop((dimx, dimy), scale=(0.7,1.), interpolation=0),
     ToLogits()])

train_data = Reader(rgb, labels, transform, transform_target)
train_loader = DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=1)

# embedding part
num_sins = args.num_sins
epsilon = args.epsilon
lr = args.lr
nepoch = args.nepoch
es_max_epoch = args.es_max_epoch # early stop max epoch
op_method = args.op_method
emb_save_name = os.path.join('./guide_functions',
                             os.path.split(train_json_path)[-1].replace('.json',f'_emb_{num_sins}.json'))
sins_list = []

for i in range(num_sins):
    if i < num_sins // 2:
        sins_list.append([np.random.uniform(low=0, high=50),
                          0,
                          np.random.uniform(low=0, high=2*np.pi)])
    else:
        sins_list.append([0,
                          np.random.uniform(low=0, high=50),
                          np.random.uniform(low=0, high=2*np.pi)])
sins = torch.nn.Sequential(*[AddSine(a, b, p) for a, b, p in sins_list])
weights_norm=log_weights_norm(10.) # this is only for calling, not fitting
cvppp_emb = Embedding(sins=sins, dims=[dimx,dimy], weights_norm=weights_norm, device=device)
# train the embedding
sins_in_text, errors = cvppp_emb.fit(train_loader, epsilon=epsilon, nepoch=nepoch, lr=lr, op_method=op_method, es_max_epoch=es_max_epoch, 
                                     emb_save_name=emb_save_name)