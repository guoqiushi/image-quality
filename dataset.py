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
    def __init__(self,root,transform=None):
        self.root = root
        self.transform = transform
        self.img_list = glob.glob(os.path.join(self.root,'train','*.jpg'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self,index):

        img_path = self.img_list[index]
        label = os.path.basename(img_path).split('.')[0]

        if label =='cat':
            label = 1
        else:
            label = 0


        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img,label