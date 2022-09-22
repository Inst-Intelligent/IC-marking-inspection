# -*- coding: utf-8 -*-
import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.basenet.vgg16_bn import vgg16_bn, init_weights
from collections import OrderedDict
from utils.feature_utils import *

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

# The end-to-end pipeline
class Model(nn.Module):
    
    def __init__(self):
        
        super(Model, self).__init__()
        
        self.attentionBranch = AttentionActivationBranch()
        self.directionBranch = DirectionRecognitionBranch()
        self.characterBranch = CharacterRecognitionBranch()
        

    
    def forward(self, x, samples):

        # Get the shared features and attention maps
        attention_maps, shared_features, global_features = self.attentionBranch(x)

        # RoI pooling, and get the arranged direction labels
        pooled_features, angle_labels, samples_info = feature_roi_pooling(attention_maps, shared_features, global_features, samples)
        if pooled_features is None:
            return None

        # Get the directions
        angle_out = self.directionBranch(pooled_features)
        angle_predict = F.softmax(angle_out, dim = 1)

        # RoI align, and get the arranged character labels
        aligned_features, num_char_boxes, char_labels, num_char_labels = batch_feature_roi_align(shared_features, samples_info, attention_maps, x)

        if aligned_features is None:
            return None
        
        # Get the seuqences
        char_out = self.characterBranch(aligned_features)
        char_predict = F.softmax(char_out, dim = 1)

        return angle_predict, angle_labels, char_predict, char_labels, num_char_boxes, samples_info
        

# UpConv block
class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

# Attention activation branch
class AttentionActivationBranch(nn.Module):
    def __init__(self, pretrained=False, freeze=False):
        super(AttentionActivationBranch, self).__init__()

        """ Base network """
        self.basenet = vgg16_bn(pretrained, freeze)

        """ U network """
        self.upconv1 = double_conv(1024, 512, 256)
        self.upconv2 = double_conv(512, 256, 128)
        self.upconv3 = double_conv(256, 128, 64)
        self.upconv4 = double_conv(128, 64, 32)

        self.basepool = nn.AvgPool2d(2)

        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1)
        )


        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())
        
    def forward(self, x):
#         print(x.size())
        """ Base network """
        sources = self.basenet(x)
        
        """ U network """
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)
 
        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)
        feature_ = y
        feature = self.upconv4(y)

        y = self.conv_cls(feature)
        return y.permute(0,2,3,1), feature_, sources[1]

class DirectionRecognitionBranch(nn.Module):

    def __init__(self, size_heatmap = (256, 256) ):

        super(DirectionRecognitionBranch, self).__init__()

        self.size_heatmap = size_heatmap

        self.conv1 = nn.Sequential(
            nn.Conv2d(704, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # 64* 8 * 8  -> 128 * 4 * 4
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # 128 * 4 * 4 -> 256 * 2 * 2
        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2)
        )

        # softmax
        self.cls = nn.Sequential(
            nn.Linear(1024 * 16, 4),
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 1024 * 16)
        x = self.cls(x)

        return x


class CharacterRecognitionBranch(nn.Module):

    def __init__(self):

        super(CharacterRecognitionBranch, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(195, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2)
        )

        self.cls = nn.Sequential(
            nn.Linear(1024 * 16, 37),
        )
    
    def forward(self, x):
        
        x = self.conv1(x)
        temp = x.detach()
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 1024 * 16)
        x = self.cls(x)

        return x


if __name__ == '__main__':
    model = CRAFT(pretrained=True)
    print(model)