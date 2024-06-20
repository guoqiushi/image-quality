import torch
from torchvision import transforms
from matplotlib import pyplot as plt
import glob
import torchvision
import torch.nn as nn
from PIL import Image


model = torchvision.models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load('models/model_state.pth'))
model.eval()

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def infer(img_path):
    class_map = {0:'low quality',1:'good quality'}
    # model = torchvision.models.resnet18(pretrained=False)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 2)
    # model.load_state_dict(torch.load('models/model_state.pth'))
    # model.eval()
    img = Image.open(img_path).convert('RGB')
    transform = data_transforms['val']
    img = transform(img)
    img = torch.unsqueeze(img,0)
    _,preds = torch.max(model(img),1)
    return class_map[preds.item()]

if __name__ == '__main__':
    for img in glob.glob('D:/dataset/val_quality/30cm_40_normal/*.jpg'):
        print(infer(img))