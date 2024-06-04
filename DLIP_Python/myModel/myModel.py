"""
Brief: Organizing Deep Learning Models(CNN) with PyTorch
Author: Jin Kwak/ 21900031
Created Date: 2024.05.12
This package(DLIP_Python\myModel\myModel.py) includes
1. LeNET
2. AlexNet
3. VGG
4. ResNet
 .
 .
"""


""" ## To the main ##
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using {device} device")
if torch.cuda.is_available(): print(f'Device name: {torch.cuda.get_device_name(0)}')
"""


"""
# Components of CNN
1. Convolutional Layer
2. Activation Layer
3. Pooling Layer
4. Fully Connected Layer

# If Batch Normalization, Usually placed after F.C, Conv and before Activation Function
Conv --> Batch Normalization --> ReLU --> Convolution --> Batch Normalization ...

# Regularization
1. L2 Regularization
2. L1 Regularization
3. Dropout/ Batch Normalization ...
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt



class LeNET(nn.Module):
    def __init__(self):
        super(LeNET, self).__init__()
        self.flatten = nn.Flatten()
        # This allows 3-Dimensional Channel
        self.conv_layer = nn.sequantial(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            # S2
            nn.MaxPool2d(2, 2),
            # C3
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            # S4
            nn.MaxPool2d(2, 2)
        )

        # Classifier
        self.fc_layers = nn.Sequential(
            # F5
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            # F6
            nn.Linear(120, 84),
            nn.ReLU(),
            # OUTPUT
            nn.Linear(84, 10)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        # Converting multidimensional data to one dimension for FC operation
        x = self.flatten(x)
        # Classification
        logit = self.fc_layers(x)

        return logit

class AlexNET(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNET, self).__init__()
        self.flatten = nn.Flatten()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3,96,11,4),
            nn.ReLU(),
            nn.Conv2d(96,256,5,1),
        )

        self.fc_layers =nn.Sequential(
            nn.Linear(256*6*6,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,num_classes)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        # Converting multidimensional data to one dimension for FC operation
        x = self.flatten(x)
        # Classification
        logit = self.fc_layers(x)

        return logit


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.flatten = nn.Flatten()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 1
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 3
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 5
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # 7
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 7
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 7
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 10
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 10
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 10
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000),
            nn.ReLU(inplace=True),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.flatten(x)
        logit = self.fc_layer(x)
        return logit


# BasicBlock class defines the building block for ResNet
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down_sampling=None, stride=1):
        super().__init__()

        self.expansion = 4  # Expansion ratio for ResNet-50, 101, 152
        self.down_sampling = down_sampling
        self.stride = stride
        self.flatten = nn.Flatten()

        self.ReLU = nn.ReLU(inplace=True)
        self.conv_layer1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv_layer2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv_layer3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0,
                                     bias=False)
        self.batch_norm3 = nn.BatchNorm2d(
            out_channels * self.expansion)  # Channels 64--> 256, 128 --> 512, 256 --> 1024

        if down_sampling:
            self.down_sampling = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, padding=0,
                          bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        # else self.downsampling is predefined as None!

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # For Feedforward
        identity = x.clone()

        out = self.conv_layer1(x)
        out = self.batch_norm1(out)
        out = self.ReLU(out)

        out = self.conv_layer2(out)
        out = self.batch_norm2(out)

        out = self.conv_layer3(out)
        out = self.batch_norm3(out)

        # Layer Change
        if self.down_sampling:  # Skip Connect
            identity = self.down_sampling(identity)

        out += identity
        out = self.ReLU(out)

        return out


# ResNet class defines the entire ResNet-50 architecture
"""
ResNet model
@Parameter:
1. block :(dtype)Class BasicBlock
2. layers:(dtype)List  Number of Iterations(?) per layer
3. image_channels:(dtype) Int Number of channels of input image
4. num_classes:(dtype) Int Number of classification classes
"""
class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64  # Initial input channels
        self.expansion = 4  # Expansion ratio for ResNet-50, 101, 152

        # conv2d, batch_norm2d, relu, maxpool2d
        self.conv = nn.Conv2d(image_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm = nn.BatchNorm2d(self.in_channels)
        self.ReLU = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # The main layers of ResNet (using self._make_layer)
        self.layer1 = self._make_layer(block, layers[0], 64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)

        # Adaptive average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer
        self.fc = nn.Linear(512 * self.expansion, num_classes)

    def forward(self, x):
        # First conv layer -> bn -> relu -> maxpooling
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.ReLU(x)
        x = self.max_pool(x)

        # Layer 1 ~ 4
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Adaptive average pooling
        x = self.avg_pool(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Fully connected layer
        x = self.fc(x)

        return x

    # _make_layer method constructs the layers for ResNet
    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        down_sampling = None
        layers = []

        # Downsample identity if we change input dimensions or channels
        if stride != 1 or self.in_channels != out_channels * self.expansion:
            down_sampling = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

        # append block layers
        layers.append(block(self.in_channels, out_channels, down_sampling, stride))

        # Expansion size is always 4 for ResNet-50, 101, 152 (e.g. 64 -> 256)
        self.in_channels = out_channels * self.expansion

        # Add additional blocks
        for idx in range(1, num_residual_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)


class SimpleVGG(nn.Module):
    def __init__(self):
        super(SimpleVGG, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),  # Adjust the input size based on your input image size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)  # Assuming 10 classes for classification
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        logit = self.fc_layers(x)
        return logit

