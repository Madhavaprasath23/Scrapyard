import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50,ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
from torch import einsum
from einops import rearrange, repeat


class CNN_BackBone(nn.Module):
    def __init__(self, backbone, pretrained=True):
        super().__init__()
        self.backbone = backbone
        self.pretrained = pretrained

        if self.backbone=='resnet18':
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT if self.pretrained else None)
        elif self.backbone=='resnet34':
            self.model = resnet34(weights=ResNet34_Weights.DEFAULT if self.pretrained else None)
        elif self.backbone=='resnet50':
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT if self.pretrained else None)
        
        self.model = nn.Sequential(*list(self.model.children())[:-2])


    def forward(self,x):
        return self.model(x)

class transposeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        nn.init.kaming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv.bias, 0)
         
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConvTransposeModule(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size, stride, padding):
        
        super().__init__()
        self.relu=nn.GELU()

        # 512 -> 256
        self.conv1 = transposeConv(in_channels, output_channels, kernel_size, stride, padding)

        # 256 -> 128

        self.conv2 = transposeConv(output_channels, output_channels//2, kernel_size, stride, padding)

        # 128 -> 64

        self.conv3 = transposeConv(output_channels//2, output_channels//4, kernel_size, stride, padding)

        # 64 -> 32

        self.conv4 = transposeConv(output_channels//4, output_channels//8, kernel_size, stride, padding)

        # 32 -> 3

        self.conv5 = transposeConv(output_channels//8, 3, kernel_size, stride, padding)



    def forward(self, x):
        x = self.conv1(x)
        x=self.relu(x)
        x = self.conv2(x)
        x=self.relu(x)
        x = self.conv3(x)
        x=self.relu(x)
        x = self.conv4(x)
        x=self.relu(x)
        x = self.conv5(x)
        return x

class CNN_BACK_BONE(nn.Module):

    def __init__(self, backbone, convtranspose):
        
        super().__init__()
        self.backbone = backbone
        self.convtranspose = convtranspose
        self.norm=nn.BatchNorm2d(3)
    
    def forward(self, x):
        first=x
        x=self.backbone(x)
        x=self.convtranspose(x)

        return self.norm(first+x)

class Make_CNN_BACK_BONE(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=CNN_BACK_BONE(CNN_BackBone('resnet18'), ConvTransposeModule(512, 512, 4, 2, 1))
    
    def forward(self, x):
        return self.model(x)
