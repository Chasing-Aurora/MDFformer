import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Tuple
from einops import rearrange
from timm.models.layers import trunc_normal_
import math

# Conv-Attention Module
class Attention(nn.Module):
    
    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim = dim  
        self.kernel_size = kernel_size  

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=2, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

        self.value_embed = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

        factor = 4

        self.attention_embed = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim // factor, 1, bias=False),
            nn.BatchNorm2d(2 * dim // factor),
            nn.ReLU(),
            nn.Conv2d(2 * dim // factor, kernel_size * kernel_size * dim, 1)
        )

    def forward(self, x):

        bs, c, h, w = x.shape  
        k1 = self.key_embed(x) 
        v = self.value_embed(x).view(bs, c, -1)  

        y = torch.cat([k1, x], dim=1)  
        att = self.attention_embed(y)  
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)  
        k2 = F.softmax(att, dim=-1) * v  
        k2 = k2.view(bs, c, h, w) 

        return k1 + k2  

# Head FPNs
class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, d, g, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True)
        )
    
class FPNHead(nn.Module):
    """Panoptic Feature Pyramid Networks
    https://arxiv.org/abs/1901.02446
    """
    def __init__(self, in_channels, channel=128, num_classes=19):
        super().__init__()
        self.lateral_convs = nn.ModuleList([])
        self.output_convs = nn.ModuleList([])

        for ch in in_channels[::-1]:
            self.lateral_convs.append(ConvModule(ch, channel, 1))
            self.output_convs.append(ConvModule(channel, channel, 3, 1, 1))

    def forward(self, features) -> Tensor:
        features = features[::-1]
        out = self.lateral_convs[0](features[0])
        
        for i in range(1, len(features)):
            out = F.interpolate(out, scale_factor=2.0, mode='nearest')
            out = out + self.lateral_convs[i](features[i])
            out = self.output_convs[i](out)
        
        return out
    
# Differential dynamic texture Attention module

class FusionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.d_conv = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)
        self.att = Attention(in_channels)

        # texture conv
        self.t_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1)
        self.t_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=3, dilation=3)
        self.t_conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=5, dilation=5)

        self.m_conv1 = nn.Conv2d(in_channels*3, in_channels*3 // 8, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.m_conv2 = nn.Conv2d(in_channels*3 // 8, in_channels*3, kernel_size=1)


    def forward(self, x1, x2):

        cat_feature = torch.cat([x1, x2], dim=1)
        d_score = self.d_conv(cat_feature)
        diff = d_score

        # multi-scale texture feature local
        t_d_1 = self.t_conv1(diff) 
        t_d_2 = self.t_conv2(diff) 
        t_d_3 = self.t_conv3(diff) 

        t_d = torch.cat([t_d_1, t_d_2, t_d_3], dim=1) #(B, C*3, H, W)
        t_d_res = t_d.clone()

        m_t_d = self.m_conv1(t_d)
        m_t_d = self.relu(m_t_d)
        m_t_d = self.m_conv2(m_t_d)
        m_t_d = torch.sigmoid(m_t_d)

        t_d = t_d_res * m_t_d
        
        # split and sum operation
        t_d_1 = t_d[:, :t_d.shape[1]//3, :, :]
        t_d_2 = t_d[:, t_d.shape[1]//3:t_d.shape[1]//3*2, :, :]
        t_d_3 = t_d[:, t_d.shape[1]//3*2:, :, :]

        mt_d = t_d_1 + t_d_2 + t_d_3

        # out = d_score * g_diff + mt_d * (1 - d_score)
        out = mt_d
        return out


# Feature Rectify Module
class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
                    nn.Linear(self.dim * 4, self.dim * 4 // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.dim * 4 // reduction, self.dim * 2), 
                    nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        max = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, max), dim=1) # B 4C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4) # 2 B C 1 1
        return channel_weights


class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
                    nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.dim // reduction, 2, kernel_size=1), 
                    nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1) # B 2C H W
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4) # 2 B 1 H W
        return spatial_weights


class FeatureRectifyModule(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5):
        super(FeatureRectifyModule, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x1, x2):
        channel_weights = self.channel_weights(x1, x2)
        spatial_weights = self.spatial_weights(x1, x2)
        out_x1 = x1 + self.lambda_c * channel_weights[1] * x2 + self.lambda_s * spatial_weights[1] * x2
        out_x2 = x2 + self.lambda_c * channel_weights[0] * x1 + self.lambda_s * spatial_weights[0] * x1
        return out_x1, out_x2 
    
def channel_shuffle(x, groups=2):
    bat_size, channels, w, h = x.shape
    group_c = channels // groups
    x = x.view(bat_size, groups, group_c, w, h)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(bat_size, -1, w, h)
    return x


class ShuffleBlock(nn.Module):
    def __init__(self, in_c, out_c, downsample=False):
        super(ShuffleBlock, self).__init__()
        self.downsample = downsample
        half_c = out_c // 2
        if downsample:
            self.branch1 = nn.Sequential(
                # 3*3 dw conv, stride = 2
                nn.Conv2d(in_c, in_c, 3, 2, 1, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                # 1*1 pw conv
                nn.Conv2d(in_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True)
            )

            self.branch2 = nn.Sequential(
                # 1*1 pw conv
                nn.Conv2d(in_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True),
                # 3*3 dw conv, stride = 2
                nn.Conv2d(half_c, half_c, 3, 2, 1, groups=half_c, bias=False),
                nn.BatchNorm2d(half_c),
                # 1*1 pw conv
                nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True)
            )

        else:
            # in_c = out_c
            assert in_c == out_c

            self.branch2 = nn.Sequential(
                # 1*1 pw conv
                nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True),
                # 3*3 dw conv, stride = 1
                nn.Conv2d(half_c, half_c, 3, 1, 1, groups=half_c, bias=False),
                nn.BatchNorm2d(half_c),
                # 1*1 pw conv
                nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True)
            )
            self.branch2_dsm = nn.Sequential(
                # 1*1 pw conv
                nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True),
                # 3*3 dw conv, stride = 1
                nn.Conv2d(half_c, half_c, 3, 1, 1, groups=half_c, bias=False),
                nn.BatchNorm2d(half_c),
                # 1*1 pw conv
                nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True)
            )

    def forward(self, x,y):
        out = None
        if self.downsample:
            # if it is downsampling, we don't need to do channel split
            out = torch.cat((self.branch1(x), self.branch2(x)), 1)
        else:
            # channel split
            channels = x.shape[1]
            c = channels // 2
            x1 = x[:, :c, :, :]
            x2 = x[:, c:, :, :]

            y1 = y[:, :c, :, :]
            y2 = y[:, c:, :, :]

            out = torch.cat((x1, self.branch2(y2)), 1)
            out_rgb = channel_shuffle(out, 2)
            
            out_dsm = torch.cat((y1, self.branch2_dsm(x2)), 1)
            out = channel_shuffle(out, 2)
            out_dsm = channel_shuffle(out_dsm, 2)

        return out_rgb, out_dsm
    
class TimeAttention(nn.Module):
    def __init__(self,
                 channels):
        super(TimeAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        attn_channels = channels // 16
        attn_channels = max(attn_channels, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels * 2, attn_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(attn_channels),
            nn.ReLU(),
            nn.Conv2d(attn_channels, channels * 2, kernel_size=1, bias=False),
        )

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.avg_pool(x)
        y = self.mlp(x)
        B, C, H, W = y.size()
        x1_attn, x2_attn = y.reshape(B, 2, C // 2, H, W).transpose(0, 1)
        x1_attn = torch.sigmoid(x1_attn)
        x2_attn = torch.sigmoid(x2_attn)
        x1 = x1 * x1_attn + x1
        x2 = x2 * x2_attn + x2
        return x1, x2
