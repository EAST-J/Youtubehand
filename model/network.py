import torch
import torch.nn as nn
from conv.spiralconv import Spiralconv, SpiralDeblock
from model.resnet import resnet18, resnet50
from model.hrnet.config import config as hrnet_config
from model.hrnet.config import update_config as hrnet_update_config
from model.hrnet.hrnet_cls_net import HighResolutionNet, get_cls_net

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
        if isinstance(self.backbone, HighResolutionNet):
            self.en_layer = self.backbone
        else:
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
        elif 'hrnet' in backbone:
            hrnet_yaml = './model/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = './model/hrnet/hrnetv2_w64_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            basenet = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
            latent_channel = 2048
        else:
            raise Exception("Backbone Types Not supported", backbone)

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

