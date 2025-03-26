import math
import numpy as np

import torch as T
import torch.nn as L
import torch.nn.init as I
import torch.nn.functional as F


class TrunkBlock(L.Module):
    def __init__(self, feat_in, feat_out):
        super(TrunkBlock, self).__init__()
        self.conv1 = L.Conv2d(feat_in, int(feat_out*1.), kernel_size=3, stride=1, padding=1, dilation=1)
        self.drop1 = L.Dropout2d(p=0.5, inplace=False)
        self.bn1 = L.BatchNorm2d(feat_in, eps=1e-05, momentum=0.25, affine=True, track_running_stats=True)

        I.xavier_normal_(self.conv1.weight, gain=I.calculate_gain('relu'))
        I.constant_(self.conv1.bias, 0.0) # current
        
    def forward(self, x):
        return F.relu(self.conv1(self.drop1(self.bn1(x))))

class PreFilter(L.Module):
    def __init__(self):
        super(PreFilter, self).__init__()
        self.conv1 = L.Sequential(
            L.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            L.ReLU(inplace=True),
            L.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = L.Sequential(
            L.Conv2d(64, 192, kernel_size=5, padding=2),
            L.ReLU(inplace=True)
        )        
        
    def forward(self, x):
        c1 = self.conv1(x)
        y = self.conv2(c1)
        return c1, y
    

class EncStage(L.Module):
    def __init__(self, trunk_width=64):
        super(EncStage, self).__init__()
        self.conv3  = L.Conv2d(192, 128, kernel_size=3, stride=1, padding=0)
        self.drop1  = L.Dropout2d(p=0.5, inplace=False) ##
        self.bn1    = L.BatchNorm2d(192, eps=1e-05, momentum=0.25, affine=True, track_running_stats=True) ##
        self.pool1  = L.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ##
        self.tw = int(trunk_width)
        self.conv4a  = TrunkBlock(128, 2*self.tw)
        self.conv5a  = TrunkBlock(2*self.tw, 2*self.tw)
        self.conv6a  = TrunkBlock(2*self.tw, 2*self.tw)
        self.conv4b  = TrunkBlock(2*self.tw, 2*self.tw)
        self.conv5b  = TrunkBlock(2*self.tw, 2*self.tw)
        self.conv6b  = TrunkBlock(2*self.tw, self.tw)
        ##
        I.xavier_normal_(self.conv3.weight, gain=I.calculate_gain('relu'))        
        I.constant_(self.conv3.bias, 0.0)
        
    def forward(self, x):
        c3 = (F.relu(self.conv3(self.drop1(self.bn1(x))), inplace=False))
        c4a = self.conv4a(c3)
        c4b = self.conv4b(c4a)
        c5a = self.conv5a(self.pool1(c4b))
        c5b = self.conv5b(c5a)
        c6a = self.conv6a(c5b)
        c6b = self.conv6b(c6a)

        return T.cat([c3, c4a[:,:self.tw], c4b[:,:self.tw]], dim=1), \
                T.cat([c5a[:,:self.tw], c5b[:,:self.tw], c6a[:,:self.tw], c6b], dim=1), \
                c6b
 
  

    
class Encoder(L.Module):
    def __init__(self, mu, trunk_width, use_prefilter):
        super(Encoder, self).__init__()
        self.mu = L.Parameter(T.from_numpy(mu), requires_grad=False) #.to(device)
        self.pre = PreFilter()
        self.enc = EncStage(trunk_width)
        self.use_prefilter = use_prefilter

    def forward(self, x):
        prefilter_fmaps_c1, prefilter_fmaps_y = self.pre(x - self.mu)
        encoder_fmaps_1, encoder_fmaps_2, h = self.enc(prefilter_fmaps_y)
        if self.use_prefilter == 0:
            return x, [encoder_fmaps_1, encoder_fmaps_2], h
        elif self.use_prefilter == 1:
            return x, [T.cat([prefilter_fmaps_c1, prefilter_fmaps_y], dim=1), encoder_fmaps_1, encoder_fmaps_2], h