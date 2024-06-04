
##########################################################
# Image Proccessing with Deep Learning
# DLIP Final 2024 Submission
#
# Date: 2024-5-28
#
# Author: Jin Kwak
#
# ID: 21900031
#
##########################################################

# Import Module
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt


# Select GPU or CPU for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")




data_dir = "../Exercise/data/hymenoptera_data"
input_size = 299


class Stem(nn.Module):
    def __init__(self):
        super(Stem, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32 , kernel_size=3, stride=2, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,64 ,kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 80, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(80, 192, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 256, kernel_size=3, stride=2, padding=0, bias=False),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.conv_layer(x)

class InceptionResNetA(nn.Module):
    def __init__(self):
        super(InceptionResNetA, self).__init__()
        self.conv1  = nn.Conv2d(256, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2  = nn.Sequential(
            nn.Conv2d(256,32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(32,32,kernel_size= 3, stride=1, padding=1, bias=False))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv4 =nn.Conv2d(96,256,kernel_size=1,stride=1,padding=0, bias = False)
    def forward(self, x):
        # x = nn.ReLU(x)
        forward_path1 = self.conv1(x)
        forward_path2 = self.conv2(x)
        forward_path3 = self.conv3(x)

        forward_path  = torch.cat([forward_path1, forward_path2, forward_path3], 1)
        forward_path  = self.conv4(forward_path)
        out = x + forward_path
        # out = nn.ReLU(out)
        return out
class ReductionA(nn.Module):
    def __init__(self):
        super(ReductionA, self).__init__()
        self.max_layer =  nn.MaxPool2d(kernel_size =3, stride =2)

        self.conv_layer1 = nn.Sequential(
             nn.Conv2d(256,384, kernel_size= 3, stride=2, padding= 0 , bias= False)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(256,96,kernel_size= 1, stride=1, padding=0, bias= False),
            nn.Conv2d(96, 96,kernel_size=3, stride=1, padding=1, bias= False),
            nn.Conv2d(96,256,kernel_size=3, stride=2, padding=0, bias= False),
            nn.ReLU(inplace= True)
        )
    def forward(self,x):
        x1 = self.max_layer(x)
        x2 = self.conv_layer1(x)
        x3 = self.conv_layer2(x)

        x_final = torch.cat([x1,x2,x3], 1)
        return x_final

class InceptionResNetV1(nn.Module):
    def __init__(self,num_classes= 10):
        super().__init__()
        self.stem = Stem()
        self.InceptionResNetA = InceptionResNetA()
        self.ReductionA = ReductionA()
        self.flatten = nn.Flatten()
        self.maxpool = nn.MaxPool2d(kernel_size= 3, stride= 2)
        self.fc_layers = nn.Sequential(
            nn.Linear(8*8*896, 256),
            nn.ReLU(inplace= True),
            nn.Linear(256,2),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.stem.forward(x)

        x = self.InceptionResNetA.forward(x)
        x = self.ReductionA.forward(x)
        x = self.maxpool(x)
        out = self.flatten(x)
        out = self.fc_layers(out)

model = InceptionResNetV1(num_classes=2).to(device)

from torchsummary import summary
summary(model, (3, 299, 299))

