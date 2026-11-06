import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
import numpy as np
from .utils import *
from .hrnet_model import HRnet_Backbone # type: ignore
from .mit import *
from .dual_mit import *

class HRCDNet(nn.Module):
    def __init__(self, in_chans=3, num_classes=2):
        super(HRCDNet, self).__init__()

        self.backbone_name = 'dual_mit_b2'

        if self.backbone_name == 'hrnetv2_w18':
            chans = [18, 36, 72, 144]
            self.backbone_rgb = HRnet_Backbone(backbone = self.backbone_name,in_channels=3, pretrained = True)
            self.backbone_dsm = HRnet_Backbone(backbone = self.backbone_name,in_channels=1, pretrained = True)
        elif self.backbone_name == 'hrnetv2_w32':
            chans = [32, 64, 128, 256]
            self.backbone_rgb = HRnet_Backbone(backbone = self.backbone_name,in_channels=3, pretrained = True)
            self.backbone_dsm = HRnet_Backbone(backbone = self.backbone_name,in_channels=1, pretrained = True)
        elif self.backbone_name == 'hrnetv2_w48':
            chans = [48, 96, 192, 384]
            self.backbone_rgb = HRnet_Backbone(backbone = self.backbone_name,in_channels=3, pretrained = True)
            self.backbone_dsm = HRnet_Backbone(backbone = self.backbone_name,in_channels=1, pretrained = True)
        elif self.backbone_name == 'mit_b2':
            chans = [64, 128, 320, 512]
            self.backbone_rgb = MiT('B2', in_channels=3)
            self.backbone_dsm = MiT('B2', in_channels=1)
            dict = torch.load('segformers/mit_b2.pth', map_location='cpu')
            # ignore patch embed1.proj.weight and patch embed1.proj.bias
            for k in list(dict.keys()):
                if 'patch_embed1.proj' in k:
                    del dict[k]
            self.backbone_rgb.load_state_dict(dict, strict=False)
            self.backbone_dsm.load_state_dict(dict, strict=False)
        elif self.backbone_name == 'dual_mit_b2':
            chans = [64, 128, 320, 512]
            self.backbone = mit_b2()
            # load_dualpath_model(self.backbone, 'segformers/mit_b2.pth')
        else:
            raise ValueError('Invalid backbone name. Please choose from hrnetv2_w18, hrnetv2_w32, or hrnetv2_w48.')

        last_inp_channels = 128

        # DDTA module
        self.ddta_1 = FusionModule(in_channels=chans[0])
        self.ddta_2 = FusionModule(in_channels=chans[1])
        self.ddta_3 = FusionModule(in_channels=chans[2])
        self.ddta_4 = FusionModule(in_channels=chans[3])

        # uphead
        self.head = FPNHead(in_channels = chans, channel=last_inp_channels, num_classes = num_classes)

        # deep supervision for seg
        self.conv_seg_1 = nn.Conv2d(in_channels=chans[0], out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        self.conv_seg_2 = nn.Conv2d(in_channels=chans[1], out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        self.conv_seg_3 = nn.Conv2d(in_channels=chans[2], out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        self.conv_seg_4 = nn.Conv2d(in_channels=chans[3], out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        self.last_seg = nn.Conv2d(in_channels=last_inp_channels, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(self, x1,x2):
        H, W = x1.size(2), x1.size(3)
        # copy x2 from 1 channel to 3 channel
        x2 = x2.expand(-1, 3, -1, -1)
        # backbone
        out_x1, out_x2 = self.backbone(x1, x2)

        # diff feature
        out = []
        out.append(self.ddta_1(out_x1[0], out_x2[0]))
        out.append(self.ddta_2(out_x1[1], out_x2[1]))
        out.append(self.ddta_3(out_x1[2], out_x2[2]))
        out.append(self.ddta_4(out_x1[3], out_x2[3]))    

        # deep supervision for seg
        out_seg_1 = self.conv_seg_1(out[0])
        out_seg_2 = self.conv_seg_2(out[1])
        out_seg_3 = self.conv_seg_3(out[2])
        out_seg_4 = self.conv_seg_4(out[3])

        x = self.head(out)

        out_seg = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        out_seg = self.last_seg(self.dropout(out_seg))


        return out_seg, out_seg_1, out_seg_2, out_seg_3, out_seg_4
        # return out_seg
    
if __name__ == '__main__':

    model = HRCDNet(in_chans=3, num_classes=2)

    # caculate flops1
    from thop import profile
    
    import time
    start_time = time.time()
    input1 = torch.randn(1, 3, 256, 256)
    input2 = torch.randn(1, 1, 256, 256)
    # res = model(input1, input2)
    end_time = time.time()
    print("time=s", end_time - start_time)
    flops1, params1 = profile(model, inputs=(input1,input2))
    print("flops=G", flops1 / (1000 ** 3))
    print("parms=M", params1 / (1000 ** 2))
    param_num = 0