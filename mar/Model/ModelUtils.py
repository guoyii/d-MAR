'''
Description: Functions od model 
Author: GuoYi
Date: 2021-06-14 21:01:21
LastEditTime: 2021-06-27 09:20:15
LastEditors: GuoYi
'''

import torch.nn as nn
import torch.nn.functional as F

## Freeze 
## ----------------------------------------
def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False

## AE_Conv
## ----------------------------------------
class AE_Down(nn.Module):
    '''
    input:N*C*H*W
    '''
    def __init__(self, in_channels, out_channels):
        super(AE_Down, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


## ----------------------------------------
class AE_Up(nn.Module):
    '''
    input:N*C*H*W
    '''
    def __init__(self, in_channels, out_channels):
        super(AE_Up, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


## Out Conv
## ----------------------------------------
class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(1, 1))

    def forward(self, x):
        return self.conv(x)



## New Model Utils
## ----------------------------------------
class basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(basic_block, self).__init__()

        self.filter1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu1 = nn.ReLU()
        self.filter2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu2 = nn.ReLU()

    def forward(self, input):
        output = self.relu1(self.filter1(input))
        output = self.filter2(output)
        output += input
        output = self.relu2(output)

        return output



## Convolution Four
## ----------------------------------------
class ResBasic(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3):
        super(ResBasic, self).__init__()
        self.ResNet = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k_size, padding=int(k_size/2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),

            nn.Conv2d(out_channels, out_channels, kernel_size=k_size, padding=int(k_size/2)),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.ResNet(x)
        out = out + self.shortcut(x)
        return F.relu(out)


