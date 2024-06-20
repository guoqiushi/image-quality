import os
import pandas as pd
import warnings
from tqdm import  tqdm
from PIL import Image
import torch
import glob
import torchvision
from torch.utils.data import Dataset, DataLoader,random_split
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

class QualityDataset(Dataset):
    def __init__(self,mode,transform=None):
        # self.root = root
        self.transform = transform
        if mode == 'train':
            self.img_list = glob.glob('D:/dataset/cityscape/*/leftImg8bit/*/*/*.png')+\
                            glob.glob('D:/dataset/cityscape/*/leftImg8bit_rain/train/*/*.png')+\
                            glob.glob('D:/dataset/cityscape/*/leftImg8bit_foggy/train/*/*.png')
        else:
            self.img_list = glob.glob('D:/dataset/val_quality/*/*.jpg')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self,index):

        img_path = self.img_list[index]
        img_name = os.path.basename(img_path).split('.')[0]
        if 'rain' in img_name or 'foggy' in img_name:
            label = 0
        else:
            label = 1
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img,label




if __name__ == '__main__':
    valdata = QualityDataset(mode='val')
    print(valdata.__getitem__(10))

