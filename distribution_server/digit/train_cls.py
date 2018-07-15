import os
import argparse
import tqdm
from itertools import chain
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable

from utils import DigitImage
from model import resnet64
from model import get_cls_loss

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default="/home/xurj/DA_Sphere/visualization/digit/data")
parser.add_argument("--source", default="SVHN")
parser.add_argument("--target", default="MNIST")
parser.add_argument("--batch_size", default=256)
parser.add_argument("--shuffle", default=True)
parser.add_argument("--num_workers", default=8)
parser.add_argument("--epoches", default=90)
parser.add_argument("--snapshot", default="/home/xurj/DA_Sphere/visualization/digit/snapshot")
parser.add_argument("--lr", default=0.1)
parser.add_argument("--momentum", default=0.9)
parser.add_argument("--weight_decay", default=0.0005)
parser.add_argument("--gpu_id", default=0)
parser.add_argument("--log_interval", default=20)
parser.add_argument("--vis_dims", default=2)
parser.add_argument("--class_num", default=10)
parser.add_argument("--extract", default=False)
args = parser.parse_args()

source_root = os.path.join(args.data_root, args.source, "imgs")
source_label = os.path.join(args.data_root, args.source, "label25000.txt")
data_transform = transforms.Compose([
    transforms.Scale((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
source_set = DigitImage(source_root, source_label, data_transform)
assert len(source_set) == 25000
source_loader_raw = torch.utils.data.DataLoader(source_set, batch_size=args.batch_size,
    shuffle=args.shuffle, num_workers=args.num_workers)

net = resnet64(vis_dims=args.vis_dims, class_num=args.class_num, extract=args.extract).cuda(args.gpu_id)
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

def print_log(epoch, i, lr, loss):
    print "Epoch [%d/%d] Iter [%d] lr: %f, source_cls_loss: %.4f" \
          % (epoch, args.epoches, i, lr, loss)


for epoch in range(args.epoches):
    source_loader = iter(source_loader_raw)

    if epoch in [30, 60]:
        args.lr *= 0.1
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
 
    for i, (source_imgs, source_labels) in tqdm.tqdm(enumerate(source_loader)):
        source_imgs, source_labels = Variable(source_imgs.cuda(args.gpu_id)), Variable(source_labels.cuda(args.gpu_id))
        
        optimizer.zero_grad()
        source_pred = net(source_imgs)
        source_cls_loss = get_cls_loss(source_pred, source_labels)
        source_cls_loss.backward()
        optimizer.step()

        if (i+1) % args.log_interval == 0:
            print_log(epoch+1, i+1, args.lr, source_cls_loss.data[0])

    torch.save(net.state_dict(), os.path.join(args.snapshot, "SVHN_Source_only_resnet64_" + str(epoch) + ".pth"))
