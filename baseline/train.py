import numpy as np
from glob import glob
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
import random
import time
import itertools
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from utils import *
from torch.autograd import Variable
from IPython.display import clear_output

from networks.HRCDNet import HRCDNet
try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from pynvml import *
import warnings
warnings.filterwarnings("ignore")
# Set the random seed for reproducibility
np.random.seed(3407)
torch.manual_seed(3407)
torch.cuda.manual_seed_all(3407)
random.seed(3407)

net = HRCDNet(in_chans=3,num_classes=6).cuda()
# torch.nn.DataParallel(net)

params = 0
for name, param in net.named_parameters():
    params += param.nelement()
print(params)
# Load the datasets

print("training : ", train_ids)
print("testing : ", test_ids)
print("BATCH_SIZE: ", BATCH_SIZE)
print("Stride Size: ", Stride_Size)

# 写入到log文件中
dir = 'logs_potsdam/baseline_try_1'
if not os.path.exists(dir):
    os.makedirs(dir)
log_file = os.path.join(dir, 'log.txt')

# 如果存在则清空文件
if os.path.exists(log_file):
    os.remove(log_file) 

log_files = open(log_file, 'a')
log_files.write("training : " + str(train_ids) + "\n")
log_files.write("testing : " + str(test_ids) + "\n")
log_files.write("BATCH_SIZE: " + str(BATCH_SIZE) + "\n")
log_files.write("Stride Size: " + str(Stride_Size) + "\n")
log_files.close()
train_set = ISPRS_dataset(train_ids, cache=CACHE)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE)

base_lr = 0.001
params_dict = dict(net.named_parameters())
params = []
for key, value in params_dict.items():
    if '_D' in key:
        # Decoder weights are trained at the nominal learning rate
        params += [{'params':[value],'lr': base_lr}]
    else:
        # Encoder weights are trained at lr / 2 (we have VGG-16 weights as initialization)
        params += [{'params':[value],'lr': base_lr / 2}]

# optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
optimizer = optim.AdamW(net.parameters(), lr=base_lr, weight_decay=0.0005)
# We define the scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0)

class DiceLoss(nn.Module):
    def __init__(self, ignore_label: int = 255) -> None:
        super().__init__()
        self.ignore_label = ignore_label

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        preds = torch.softmax(preds, dim=1)
        smooth = 1e-5

        intersection = torch.sum(preds[:, 1, :, :] * (labels == 1).float())
        union = torch.sum(preds[:, 1, :, :]) + torch.sum((labels == 1).float()) + smooth

        dice = (2.0 * intersection + smooth) / union
        return 1 - dice

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([self._forward(pred, labels) for pred in preds])
        return self._forward(preds, labels)
    
class FocalLoss(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.gamma = gamma
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        loss = self.criterion(preds, labels)
        p = torch.exp(-loss)
        loss_focal = (1 - p) ** self.gamma * loss
        return torch.mean(loss_focal)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([self._forward(pred, labels) for pred in preds])
        return self._forward(preds, labels)

focal_loss_fn = FocalLoss()
dice_loss_fn = DiceLoss()

def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    # Use the network on the test set
    ## Potsdam
    if DATASET == 'Potsdam':
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, :3], dtype='float32') for id in test_ids)
        # test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, (3, 0, 1, 2)][:, :, :3], dtype='float32') for id in test_ids)
    ## Vaihingen
    else:
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_dsms = (np.asarray(io.imread(DSM_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
    eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
    all_preds = []
    all_gts = []

    # Switch the network to inference mode
    with torch.no_grad():
        for img, dsm, gt, gt_e in tqdm(zip(test_images, test_dsms, test_labels, eroded_labels), total=len(test_ids), leave=False):
            pred = np.zeros(img.shape[:2] + (N_CLASSES,))

            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(
                    tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                        leave=False)):
                # Build the tensor
                image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                image_patches = np.asarray(image_patches)
                image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)

                min = np.min(dsm)
                max = np.max(dsm)
                dsm = (dsm - min) / (max - min)
                dsm_patches = [np.copy(dsm[x:x + w, y:y + h]) for x, y, w, h in coords]
                dsm_patches = np.asarray(dsm_patches)
                dsm_patches = Variable(torch.from_numpy(dsm_patches).cuda(), volatile=True)

                # Do the inference
                dsm_patches = dsm_patches.unsqueeze(1)
                outs,out_seg_1, out_seg_2, out_seg_3, out_seg_4 = net(image_patches, dsm_patches)
                # outs = net(image_patches)
                outs = outs.data.cpu().numpy()

                # Fill in the results array
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out
                del (outs)

            pred = np.argmax(pred, axis=-1)
            all_preds.append(pred)
            all_gts.append(gt_e)
            clear_output()
            
    accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                       np.concatenate([p.ravel() for p in all_gts]).ravel())
    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy


def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=1,log_file=log_file,save_dir=dir):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.cuda()

    criterion = nn.NLLLoss2d(weight=weights)
    iter_ = 0
    acc_best = 50.0

    for e in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        net.train()
        for batch_idx, (data, dsm, target) in enumerate(train_loader):
            data, dsm, target = Variable(data.cuda()), Variable(dsm.cuda()), Variable(target.cuda())
            optimizer.zero_grad()

            # 维度1添加一个维度
            dsm = dsm.unsqueeze(1)

            output,out_seg_1,out_seg_2,out_seg_3,out_seg_4 = net(data, dsm)
            # gen multi-scale target
            target_1 = F.interpolate(target.unsqueeze(1).float(), size=out_seg_1.shape[2:], mode='nearest').squeeze(1).long()
            target_2 = F.interpolate(target.unsqueeze(1).float(), size=out_seg_2.shape[2:], mode='nearest').squeeze(1).long()
            target_3 = F.interpolate(target.unsqueeze(1).float(), size=out_seg_3.shape[2:], mode='nearest').squeeze(1).long()
            target_4 = F.interpolate(target.unsqueeze(1).float(), size=out_seg_4.shape[2:], mode='nearest').squeeze(1).long()
            # output = net(data)
            loss = CrossEntropy2d(output, target, weight=weights)
            loss_1 = CrossEntropy2d(out_seg_1, target_1, weight=weights)
            loss_2 = CrossEntropy2d(out_seg_2, target_2, weight=weights)
            loss_3 = CrossEntropy2d(out_seg_3, target_3, weight=weights)
            loss_4 = CrossEntropy2d(out_seg_4, target_4, weight=weights)
            loss = loss * 0.8 + (loss_1 + loss_2 + loss_3 + loss_4) * 0.2
            # dice loss
            # print(f"Output shape: {output.shape}, Target shape: {target.shape}")
            loss_dice = dice_loss_fn(output, target)
            loss_dice_1 = dice_loss_fn(out_seg_1, target_1)
            loss_dice_2 = dice_loss_fn(out_seg_2, target_2)
            loss_dice_3 = dice_loss_fn(out_seg_3, target_3)
            loss_dice_4 = dice_loss_fn(out_seg_4, target_4)
            loss_dice = loss_dice * 0.8 + (loss_dice_1 + loss_dice_2 + loss_dice_3 + loss_dice_4) * 0.2

            # focal loss
            loss_focal = focal_loss_fn(output, target)
            loss_focal_1 = focal_loss_fn(out_seg_1, target_1)
            loss_focal_2 = focal_loss_fn(out_seg_2, target_2)
            loss_focal_3 = focal_loss_fn(out_seg_3, target_3)
            loss_focal_4 = focal_loss_fn(out_seg_4, target_4)
            loss_focal = loss_focal * 0.8 + (loss_focal_1 + loss_focal_2 + loss_focal_3 + loss_focal_4) * 0.2

            loss = loss + loss_dice + loss_focal

            loss.backward()
            optimizer.step()

            # losses[iter_] = loss.data
            # mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])

            if iter_ % 100 == 0:
                clear_output()
                rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0], (1, 2, 0)), dtype='uint8')
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data, accuracy(pred, gt)))
                # 写入到log文件中
                log_files = open(log_file, 'a')
                log_files.write('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}\n'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data, accuracy(pred, gt)))
                log_files.close()
            iter_ += 1

            del (data, target, loss)
            weights_dir = os.path.join(save_dir, 'epoch{}.pth')
            torch.save(net.state_dict(), weights_dir.format(e))
            # # if e % save_epoch == 0:
            # if iter_ % 1000 == 0:
            #     net.eval()
            #     acc = test(net, test_ids, all=False, stride=Stride_Size)
            #     net.train()
            #     if acc > acc_best:
            #         # torch.save(net.state_dict(), './logs/deeplabv3/epoch{}_{}'.format(e, acc))
            #         weights_dir = os.path.join(save_dir, 'best_model.pth')
            #         torch.save(net.state_dict(), weights_dir.format(e, acc))
            #         acc_best = acc
    print('acc_best: ', acc_best)

# #####   train   ####
# time_start=time.time()
# train(net, optimizer, 50, scheduler,log_file=log_file,save_dir=dir)
# time_end=time.time()
# print('Total Time Cost: ',time_end-time_start)

####   test   ####
net.load_state_dict(torch.load('logs_potsdam/baseline_try_1/epoch34'))
save_dir = 'results/potsdam/e34'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
net.eval()
acc, all_preds, all_gts = test(net, test_ids, all=True, stride=64)
print("Acc: ", acc)
for p, id_ in zip(all_preds, test_ids):
    img = convert_to_color(p)
    # plt.imshow(img) and plt.show()
    io.imsave(os.path.join(save_dir, '{}.png'.format(id_)), img)
    # io.imsave('/inference_tile{}.png'.format(id_), img)
