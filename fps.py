import time
import torch.nn.init
from networks.HRCDNet import HRCDNet
from thop import profile

net = HRCDNet(in_chans=3, num_classes=6).cuda()
params = 0
for name, param in net.named_parameters():
    # 生成器，生成的是模型中每个可训练参数的名字和对应的参数（即权重或偏置）
    params += param.nelement()  # 返回一个张量 param 中所有元素的数量
print(params)
x1 = torch.randn(2, 3, 256, 256).cuda()
x2 = torch.randn(2, 1, 256, 256).cuda()
begin = time.time()
net(x1, x2)
end = time.time()
fps =  2 / (end - begin)

infos=''
flops, params = profile(net, (x1,x2))
infos += 'flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0) + '\n'
infos += f"FPS: {fps:.2f}  \n"
print( infos)

