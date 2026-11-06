#%%
import yaml
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from networks.utils import ShuffleBlock, TimeAttention

with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)
import torch
import torch.nn as nn
import math
import warnings
from torch.nn.modules.utils import _pair as to_2tuple
from bricks import DownSample, LayerScale, StochasticDepth, DWConv3x3, NormLayer

class StemConv(nn.Module):
    '''following ConvNext paper'''
    def __init__(self, in_channels, out_channels, bn_momentum=0.99):
        super(StemConv, self).__init__()

        self.proj = nn.Sequential(
                                    nn.Conv2d(in_channels, out_channels//2,
                                                kernel_size=(3,3), stride=(2,2), padding=(1,1)),
                                    NormLayer(out_channels//2, norm_type=config['norm_typ']),
                                    nn.GELU(),
                                    nn.Conv2d(out_channels//2, out_channels,
                                                kernel_size=(3,3), stride=(2,2), padding=(1,1)),
                                    NormLayer(out_channels, norm_type=config['norm_typ'])
                                )
    
    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.size()
        # x = x.flatten(2).transpose(1,2) # B*C*H*W -> B*C*HW -> B*HW*C
        return x, H, W


class FFN(nn.Module):
    '''following ConvNext paper'''
    def __init__(self, in_channels, out_channels, hid_channels):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, hid_channels, 1)
        self.dwconv = DWConv3x3(hid_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hid_channels, out_channels, 1)

    def forward(self, x):

        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.fc2(x)

        return x

class BlockFFN(nn.Module):
    def __init__(self, in_channels, out_channels, hid_channels, ls_init_val=1e-2, drop_path=0.):
        super().__init__()
        self.norm = NormLayer(in_channels, norm_type=config['norm_typ'])
        self.ffn = FFN(in_channels, out_channels, hid_channels)
        self.layer_scale = LayerScale(in_channels, init_value=ls_init_val)
        self.drop_path = StochasticDepth(p=drop_path)
    
    def forward(self, x):
        skip = x.clone()

        x = self.norm(x)
        x = self.ffn(x)
        x = self.layer_scale(x)
        x = self.drop_path(x)

        op = skip + x
        return op


class MSCA(nn.Module):

    def __init__(self, dim):
        super(MSCA, self).__init__()
        # input
        self.conv55 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim) 
        # split into multipats of multiscale attention
        self.conv17_0 = nn.Conv2d(dim, dim, (1,7), padding=(0, 3), groups=dim)
        self.conv17_1 = nn.Conv2d(dim, dim, (7,1), padding=(3, 0), groups=dim)

        self.conv111_0 = nn.Conv2d(dim, dim, (1,11), padding=(0,5), groups=dim)
        self.conv111_1 = nn.Conv2d(dim, dim, (11,1), padding=(5,0), groups=dim)

        self.conv211_0 = nn.Conv2d(dim, dim, (1,21), padding=(0, 10), groups=dim)
        self.conv211_1 = nn.Conv2d(dim, dim, (21,1), padding=(10, 0), groups=dim)

        self.conv11 = nn.Conv2d(dim, dim, 1) # channel mixer

    def forward(self, x):
        
        skip = x.clone()

        c55 = self.conv55(x)
        c17 = self.conv17_0(x)
        c17 = self.conv17_1(c17)
        c111 = self.conv111_0(x)
        c111 = self.conv111_1(c111)
        c211 = self.conv211_0(x)
        c211 = self.conv211_1(c211)

        add = c55 + c17 + c111 + c211

        mixer = self.conv11(add)

        op = mixer * skip

        return op

class BlockMSCA(nn.Module):
    def __init__(self, dim, ls_init_val=1e-2, drop_path=0.0):
        super().__init__()
        self.norm = NormLayer(dim, norm_type=config['norm_typ'])
        self.proj1 = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()
        self.msca = MSCA(dim)
        self.proj2 = nn.Conv2d(dim, dim, 1)
        self.layer_scale = LayerScale(dim, init_value=ls_init_val)
        self.drop_path = StochasticDepth(p=drop_path)
        # print(f'BlockMSCA {drop_path}')
    def forward(self, x):

        skip = x.clone()

        x = self.norm(x)
        x = self.proj1(x)
        x = self.act(x)
        x = self.msca(x)
        x = self.proj2(x)
        x = self.layer_scale(x)
        x = self.drop_path(x)

        out = x + skip

        return out


class StageMSCA(nn.Module):
    def __init__(self, dim, ffn_ratio=4., ls_init_val=1e-2, drop_path=0.0):
        super().__init__()
        # print(f'StageMSCA {drop_path}')
        self.msca_block = BlockMSCA(dim, ls_init_val, drop_path)

        ffn_hid_dim = int(dim * ffn_ratio)
        self.ffn_block = BlockFFN(in_channels=dim, out_channels=dim,
                                  hid_channels=ffn_hid_dim, ls_init_val=ls_init_val,
                                  drop_path=drop_path)

    def forward(self, x): # input coming form Stem
        # B, N, C = x.shape
        # x = x.permute()
        x = self.msca_block(x)
        x = self.ffn_block(x)

        return x

class MSCANet(nn.Module):
    def __init__(self, in_channnels=3, embed_dims=[32, 64, 460,256],
                 ffn_ratios=[4, 4, 4, 4], depths=[3,3,5,2], num_stages=4,
                 ls_init_val=1e-2, drop_path=0.0):
        super(MSCANet, self).__init__()
        # print(f'MSCANet {drop_path}')
        self.depths = depths
        self.num_stages = num_stages
        # stochastic depth decay rule (similar to linear decay) / just like matplot linspace
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        cur = 0

        for i in range(num_stages):
            if i == 0:
                input_embed = StemConv(in_channnels, embed_dims[0])
            else:
                input_embed = DownSample(in_channels=embed_dims[i-1], embed_dim=embed_dims[i])
            
            stage = nn.ModuleList([StageMSCA(dim=embed_dims[i], ffn_ratio=ffn_ratios[i],
                                   ls_init_val=ls_init_val, drop_path=dpr[cur + j])
                                   for j in range(depths[i])])

            norm_layer = NormLayer(embed_dims[i], norm_type=config['norm_typ'])
            cur += depths[i]

            setattr(self, f'input_embed{i+1}', input_embed)
            setattr(self, f'stage{i+1}', stage)
            setattr(self, f'norm_layer{i+1}', norm_layer)

        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        cur = 0
        for i in range(num_stages):
            if i == 0:
                input_embed = StemConv(in_channnels, embed_dims[0])
            else:
                input_embed = DownSample(in_channels=embed_dims[i - 1], embed_dim=embed_dims[i])

            stage = nn.ModuleList([StageMSCA(dim=embed_dims[i], ffn_ratio=ffn_ratios[i],
                                             ls_init_val=ls_init_val, drop_path=dpr[cur + j])
                                   for j in range(depths[i])])

            norm_layer = NormLayer(embed_dims[i], norm_type=config['norm_typ'])
            cur += depths[i]

            setattr(self, f'input_embed1{i + 1}', input_embed)
            setattr(self, f'stage1{i + 1}', stage)
            setattr(self, f'norm_layer1{i + 1}', norm_layer)

        chans = [embed_dims[0], embed_dims[1], embed_dims[2], embed_dims[3]]
        self.SFMs = nn.ModuleList([
            ShuffleBlock(chans[0], chans[0]),
            ShuffleBlock(chans[1], chans[1]),
            ShuffleBlock(chans[2], chans[2]),
            ShuffleBlock(chans[3], chans[3])])

        self.TAMs = nn.ModuleList([
            TimeAttention(chans[0]),
            TimeAttention(chans[1]),
            TimeAttention(chans[2]),
            TimeAttention(chans[3])])

        self.apply(self._init_weights)

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

    def forward(self, x ,x1):
        B = x.shape[0]

        outs = []
        outs_fused = []

        for i in range(self.num_stages):

            input_embed = getattr(self, f'input_embed{i+1}')
            stage = getattr(self, f'stage{i+1}')
            norm_layer = getattr(self, f'norm_layer{i+1}')

            input_embed1 = getattr(self, f'input_embed1{i + 1}')
            stage1 = getattr(self, f'stage1{i + 1}')
            norm_layer1 = getattr(self, f'norm_layer1{i + 1}')

            x, H, W = input_embed(x)
            x1, H, W = input_embed1(x1)

            for stg in stage:
                x = stg(x)
            for stg in stage1:
                x1 = stg(x1)

            x = norm_layer(x)
            x1 = norm_layer1(x1)

            x_rgb = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            x_e = x1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            x, x1 = self.TAMs[i](x_rgb, x_e)
            x_fused_rgb,x_fused_e = self.SFMs[i](x, x1)
            outs.append(x_fused_rgb)
            outs_fused.append(x_fused_e)

        return outs, outs_fused
