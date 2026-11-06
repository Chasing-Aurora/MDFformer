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

class ConvModule(nn.Sequential):  ###  经常使用！！！就是基本单元了
    '''
    继承自 nn.Sequential，即顺序容器
    用于按顺序组合多个神经网络层
    '''
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
        """
        :param in_channels:一个列表，表示来自不同层的特征图通道数
        :param channel:所有特征层最终被降维到 128 通道
        :param num_classes:
        """
        super().__init__()
        self.lateral_convs = nn.ModuleList([])  # 存放 1×1 卷积 层
        self.output_convs = nn.ModuleList([])  # 存放 3×3 卷积 层

        for ch in in_channels[::-1]:  # 逆序遍历 in_channels（因为 FPN 采用自顶向下融合）
            self.lateral_convs.append(ConvModule(ch, channel, 1)) # 存放 1×1 卷积 层
            self.output_convs.append(ConvModule(channel, channel, 3, 1, 1)) # 存放 3×3 卷积 层

    def forward(self, features) -> Tensor:
        features = features[::-1]
        out = self.lateral_convs[0](features[0])  # 最高层特征图做1×1 卷积降维

        # 逐层上采样与融合
        for i in range(1, len(features)):
            out = F.interpolate(out, scale_factor=2.0, mode='nearest')  # 上采样 2 倍
            # 跳跃连接（skip connection），让高层语义特征和低层空间信息互
            out1 = self.lateral_convs[i](features[i])  # 和当前层特征相加（跳跃连接）
            out = out + F.interpolate(out1,out.shape[2:])
            out = self.output_convs[i](out)  # 3×3 卷积进一步处理
        
        return out
    
# Differential dynamic texture Attention module

# --------------------SDEM-------------------------------  #
class FusionModule(nn.Module):
    # 将两个输入特征图融合，提取和增强纹理特征
    def __init__(self, in_channels):
        super().__init__()
        self.d_conv = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)
        # 拼接后，使用1*1卷积通道数从 in_channels*2 压缩到 in_channels
        self.att = Attention(in_channels)
        # 计算输入特征的权重，以便对不同部分的特征进行选择性增强

        # texture conv
        # 提取多尺度纹理特征 感受野变大，有助于捕获不同尺度的信息
        self.t_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1)
        self.t_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=3, dilation=3)
        self.t_conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=5, dilation=5)

        self.m_conv1 = nn.Conv2d(in_channels*3, in_channels*3 // 8, kernel_size=1)
        # 进行通道数调整
        self.relu = nn.ReLU(inplace=True)
        # 调整之后再进行恢复
        self.m_conv2 = nn.Conv2d(in_channels*3 // 8, in_channels*3, kernel_size=1)


    def forward(self, x1, x2):

        cat_feature = torch.cat([x1, x2], dim=1) # 沿着通道维度拼接
        d_score = self.d_conv(cat_feature)  # 通过 1x1 卷积压缩到in_channels
        diff = d_score   # diff 是 d_score 的副本

        # multi-scale texture feature local
        # 对 diff 特征图进行多尺度卷积，提取不同尺度的纹理特征
        t_d_1 = self.t_conv1(diff) 
        t_d_2 = self.t_conv2(diff) 
        t_d_3 = self.t_conv3(diff) 

        # 沿着通道维度拼接起来，形成一个多尺度特征图
        t_d = torch.cat([t_d_1, t_d_2, t_d_3], dim=1) #(B, C*3, H, W)
        t_d_res = t_d.clone()  # 保留 t_d 的副本，后面将用于与处理后的特征进行结合

        # 使用 m_conv1 和 m_conv2 对 t_d 进行通道调整，并使用 ReLU 激活函数和 Sigmoid 激活函数
        m_t_d = self.m_conv1(t_d)
        m_t_d = self.relu(m_t_d)      # enhance MLP
        m_t_d = self.m_conv2(m_t_d)
        m_t_d = torch.sigmoid(m_t_d)  #  将特征图的值映射到 [0, 1] 范围，表示每个位置的特征重要性（权重）

        t_d = t_d_res * m_t_d  # 副本t_d_res 和由 m_t_d 生成的权重图相乘，从而进行加权处理，选择性地增强某些特征。
        
        # split and sum operation
        # 切片操作将 t_d 分为三部分，分别对应不同的尺度特征
        # 按照 通道 来划分
        t_d_1 = t_d[:, :t_d.shape[1]//3, :, :]  # 从第0到C*3//3的通道部分
        t_d_2 = t_d[:, t_d.shape[1]//3:t_d.shape[1]//3*2, :, :]
        t_d_3 = t_d[:, t_d.shape[1]//3*2:, :, :]

        # 将 每个尺度（即不同的扩张卷积特征）分成单独的部分
        # 将三个部分的特征加在一起，得到融合后的特征
        mt_d = t_d_1 + t_d_2 + t_d_3

        # out = d_score * g_diff + mt_d * (1 - d_score)
        out = mt_d
        return out


# --------------------CSFM-2-------------------------------  #
## !!!!!!
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


# !!!!!
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

# --------------------CSFM-------------------------------  #
def channel_shuffle(x, groups=2):
    bat_size, channels, w, h = x.shape
    group_c = channels // groups    ## 把cahnnel分组
    x = x.view(bat_size, groups, group_c, w, h)    # 变成五维
    # 这一步将输入的通道数分成 groups 组，每组的通道数为 group_c，并保留原始图像的宽高维度
    x = torch.transpose(x, 1, 2).contiguous()
    # 第二维（groups）和第三维（group_c）进行交换。换句话说，它将每个通道组的顺序进行打乱
    # 为什么呢？？因为相乘的结果肯定不变的，只是不同划分方式了而已
    # Channel 0: [0, 4]  (即原来属于第0组的通道0和属于第1组的通道0)
    x = x.view(bat_size, -1, w, h)
    # 恢复为原来，-1 表示该维度会根据其它维度自动推算出大小
    return x


class ShuffleBlock(nn.Module):
    # 结合了 深度可分离卷积和通道混洗（Channel Shuffle）的操作
    def __init__(self, in_c, out_c, downsample=False):
        super(ShuffleBlock, self).__init__()
        self.downsample = downsample
        half_c = out_c // 2
        # 模块的初始化分为两种情况：是否执行下采样操作
        if downsample:
            self.branch1 = nn.Sequential(
                # 3*3 dw conv, stride = 2  下采样特征图
                #  groups=in_c 表示深度卷积，每个输入通道只与自己的卷积核进行卷积
                nn.Conv2d(in_c, in_c, 3, 2, 1, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                # 1*1 pw conv  通道数减少为 half_c
                nn.Conv2d(in_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True)
            )
            # branch2 也涉及到两个卷积操作
            """
            但是这里的卷积操作先通过 1x1 点卷积减小通道数，
            然后使用 3x3 深度卷积进行下采样，最后再次通过 1x1 点卷积恢复通道数。
            """
            self.branch2 = nn.Sequential(
                # 1*1 pw conv  减小通道数
                nn.Conv2d(in_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True),
                # 3*3 dw conv, stride = 2  下采样
                nn.Conv2d(half_c, half_c, 3, 2, 1, groups=half_c, bias=False),
                nn.BatchNorm2d(half_c),
                # 1*1 pw conv  恢复通道数
                nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True)
            )

        else:    # 非下采样
            # in_c = out_c
            assert in_c == out_c
            # 分支 2 和分支 1 共享相似的结构，只不过此时不执行下采样（s=1），保持空间大小不变

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
        """
        :param x: 形状是 (batch_size, channels, height, width)
        :param y: 与 x 形状相同
        """
        out = None
        if self.downsample:
            '''
            branch1(x) 和 branch2(x) 分别进行处理后，
            使用 torch.cat 将它们在通道维度（dim=1）拼接在一起，形成一个更大的输出张量。
            '''
            # if it is downsampling, we don't need to do channel split
            out = torch.cat((self.branch1(x), self.branch2(x)), 1)
        else:
            # channel split   将输入 x 和 y 分成两部分（各自的前半部分和后半部分）
            channels = x.shape[1]
            c = channels // 2
            x1 = x[:, :c, :, :]
            x2 = x[:, c:, :, :]

            y1 = y[:, :c, :, :]
            y2 = y[:, c:, :, :]

            # 将 x1 和 branch2(y2) 拼接
            out = torch.cat((x1, self.branch2(y2)), 1)
            out_rgb = channel_shuffle(out, 2)
            # 将 y1 和 branch2_dsm(x2)
            out_dsm = torch.cat((y1, self.branch2_dsm(x2)), 1)
            # out = channel_shuffle(out, 2)
            out_dsm = channel_shuffle(out_dsm, 2)
            # 每次拼接后，我们通过 channel_shuffle(out, 2) 来打乱通道
        return out_rgb, out_dsm
    
class TimeAttention(nn.Module):
    def __init__(self,
                 channels):
        super(TimeAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 将输入的每个通道的 height 和 width 压缩为 1，-> (B, C, 1, 1)
        attn_channels = channels // 16  # 计算注意力通道数
        attn_channels = max(attn_channels, 8)
        # 保证 attn_channels 至少为 8，避免通道数过小导致无法学习到有效的注意力特征
        self.mlp = nn.Sequential(
            # 将输入的通道数从 channels * 2 转换为 attn_channels
            nn.Conv2d(channels * 2, attn_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(attn_channels),
            nn.ReLU(),
            nn.Conv2d(attn_channels, channels * 2, kernel_size=1, bias=False),
            ## 将其转换回 channels * 2
        )

    def forward(self, x1, x2):
        # 形状都是 (B, C, H, W)
        x = torch.cat((x1, x2), dim=1)
        x = self.avg_pool(x)  # (B, 2*C, 1, 1)
        y = self.mlp(x)
        B, C, H, W = y.size()
        x1_attn, x2_attn = y.reshape(B, 2, C // 2, H, W).transpose(0, 1)
        # 分组，打乱，类似shuffle
        # x1_attn：与 x1 相关的注意力权重。
        # x2_attn：与 x2 相关的注意力权重。
        x1_attn = torch.sigmoid(x1_attn)
        # 使用 sigmoid 激活函数将注意力权重映射到 [0, 1] 范围内，这样它们可以作为加权因子
        x2_attn = torch.sigmoid(x2_attn)
        x1 = x1 * x1_attn + x1
        x2 = x2 * x2_attn + x2
        return x1, x2
