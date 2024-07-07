from __future__ import absolute_import, division, print_function
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet_encoder import *
from .PoseResNet import PoseDecoder

from .cehrlf_encoder import FasterNet
from .DispResNet import DepthDecoder




class DispPoseResNet(nn.Module):
    def __init__(self, num_layers = 18, pretrained = True, num_channel=1, num_pose_frames=2,scales=range(4)):
        super(DispPoseResNet, self).__init__()
        self.num_pose_frames = num_pose_frames
        self.scales = scales
        self.num_layers = num_layers

        self.depth_encoder = FasterNet(in_chans=1)   #1
        self.depth_decoder = DepthDecoder(self.depth_encoder.num_ch_enc)

        self.pose_encoder = ResnetEncoder(num_layers=num_layers, pretrained=pretrained, num_input_images=1, num_channel=num_channel)
        self.pose_decoder = PoseDecoder(self.pose_encoder.num_ch_enc*2)

        self.DispResNet = DispResNet(self.depth_encoder, self.depth_decoder)
        self.PoseResNet = PoseResNet(self.depth_encoder, self.pose_decoder)


    def init_weights(self):
        pass


class DispResNet(nn.Module):

    def __init__(self, encoder, decoder):
        super(DispResNet, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def init_weights(self):
        pass

    def forward(self, x):
        enc_features = self.encoder(x)
        outputs = self.decoder(enc_features)

        if self.training:
            return outputs
        else:
            return outputs[0]


class PoseResNet(nn.Module):

    def __init__(self, encoder, decoder):
        super(PoseResNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def init_weights(self):
        pass

    def forward(self, img1, img2):
        features1 = self.encoder(img1)
        features2 = self.encoder(img2)

        features = []
        for k in range(0, len(features1)) : 
            features.append(torch.cat([features1[k],features2[k]],dim=1))
        pose = self.decoder([features])
        return pose

