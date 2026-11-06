import torch
import yaml
with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)
from model import SegNext

model = SegNext(num_classes=20, in_channnels=3, embed_dims=[32, 64, 460, 256],
                ffn_ratios=[4, 4, 4, 4], depths=[3, 3, 5, 2], num_stages=4,
                dec_outChannels=256, drop_path=0.2,
                config=config).cuda()
x =  torch.randn((2,3,256,256)).cuda()
o =  model(x)
print(o.shape)
