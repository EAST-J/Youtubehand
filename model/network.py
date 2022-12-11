import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from conv.spiralconv import Spiralconv
from .resnet import resnet18, resnet50
import numpy as np


def Pool(x, trans, dim=1):
    """
    x:input feature
    trans:sample matrix
    dim:sample dim
    return:sampled feature
    """
    trans = trans.to(x.device)
    row, col = trans._indices()
    value = trans._values().unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value
    out = scatter_add(out, row, dim, dim_size=trans.size(0))
    return out


class SpiralDeblock(nn.Module):
    # Decoder that include upsampling and GCN
    def __init__(self, in_channels, out_channels, indices):
        super().__init__()
        self.conv = Spiralconv(in_channels=in_channels, out_channels=out_channels, indices=indices)

    def forward(self, x, up_transform):
        out = Pool(x, up_transform)
        out = F.relu(self.conv(out))

        return out



class ResNetEncoder(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = backbone.fc

    def forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x))) #
        x = self.maxpool(x0)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.avgpool(x4)
        x = torch.flatten(x, 1)
        return x



class Network(nn.Module):  # Hand Reconstruction network
    def __init__(self, in_channels, out_channels, spiral_indices, up_transform, down_transform, backbone='resnet18'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spiral_indices = spiral_indices
        self.up_transform = up_transform
        self.down_transform = down_transform
        self.backbone, self.latent_size = self.get_backbone(backbone)
        self.num_vert = [u.size(0) for u in self.up_transform] + [self.up_transform[-1].size(1)]
        # encoder
        self.en_layer = ResNetEncoder(self.backbone)
        # decoder
        self.de_layers = nn.ModuleList()
        self.de_layers.append(
            nn.Linear(self.latent_size, self.num_vert[-1] * self.out_channels[-1])
        )
        for idx in range(len(out_channels)):
            if idx == 0:
                self.de_layers.append(
                    SpiralDeblock(out_channels[-idx - 1],
                                  out_channels[-idx - 1],
                                  self.spiral_indices[-idx - 1]))
            else:
                self.de_layers.append(
                    SpiralDeblock(out_channels[-idx], out_channels[-idx - 1],
                                  self.spiral_indices[-idx - 1]))
        self.de_layers.append(
            Spiralconv(out_channels[0], in_channels, self.spiral_indices[0]))
        # pred camera parameters
        self.cam_pred_1 = nn.Linear(778, 150)
        self.cam_pred_2 = nn.Linear(150, 3)
        self.cam_pred_3 = nn.Linear(3, 1)

    def encoder(self, x):
        return self.en_layer(x)

    def decoder(self, x):
        layer_nums = len(self.de_layers)
        feature_nums = layer_nums - 2
        for i, layer in enumerate(self.de_layers):
            if i == 0:
                x = layer(x)
                x = x.view((-1, self.num_vert[-1], self.out_channels[-1])) # reshape it into the [B, N, C], 2D feature mapping to 3D
            elif i != layer_nums - 1:
                x = layer(x, self.up_transform[feature_nums - i])
            else:
                x = layer(x)
        return x

    def get_backbone(self, backbone, pretrained=True):
        if '50' in backbone:
            basenet = resnet50(pretrained=pretrained)
            latent_channel = 2048
        elif '18' in backbone:
            basenet = resnet18(pretrained=pretrained)
            latent_channel = 512
        else:
            raise Exception("Backbone Types Not supported, Please refer to resnet.py", backbone)

        return basenet, latent_channel

    def forward(self, x):
        x1 = self.encoder(x)
        output = self.decoder(x1) # B*778*3
        # pred cam
        cam_pred = output.transpose(1, 2)
        cam_pred = self.cam_pred_1(cam_pred) # B*3*150
        cam_pred = self.cam_pred_2(cam_pred) # B*3*3
        cam_pred = self.cam_pred_3(cam_pred) # B*3*1
        cam_pred = cam_pred.squeeze(-1)
        return {'pred_vertices': output,
                'pred_camera': cam_pred,
                }

