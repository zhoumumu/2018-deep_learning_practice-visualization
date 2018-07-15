import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable

from utils import DigitImage
from model import resnet64

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default="/home/xurj/DA_Sphere/visualization/digit/data")
parser.add_argument("--t", default="SVHN")
parser.add_argument("--batch_size", default=256)
parser.add_argument("--shuffle", default=False)
parser.add_argument("--num_workers", default=8)
parser.add_argument("--snapshot", default="/home/xurj/DA_Sphere/visualization/digit/snapshot")
parser.add_argument("--epoch", default=0)
parser.add_argument("--gpu_id", default=0)
parser.add_argument("--result", default="/home/xurj/DA_Sphere/visualization/digit/result")
parser.add_argument("--vis_dims", default=2)
parser.add_argument("--class_num", default=10)
parser.add_argument("--extract", default=False)
args = parser.parse_args()

result = open(os.path.join(args.result, "resnet64_" + args.t + "_score.txt"), "a")

t_root = os.path.join(args.data_root, args.t, "imgs")
t_label = os.path.join(args.data_root, args.t, "label9000.txt")
data_transform = transforms.Compose([
    transforms.Scale((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
t_set = DigitImage(t_root, t_label, data_transform)
assert len(t_set) == 9000
t_loader = torch.utils.data.DataLoader(t_set, batch_size=args.batch_size,
    shuffle=args.shuffle, num_workers=args.num_workers)

net = resnet64(vis_dims=args.vis_dims, class_num=args.class_num, extract=args.extract).cuda(args.gpu_id)
net.eval()
net.load_state_dict(torch.load(os.path.join(args.snapshot, "SVHN_Source_only_resnet64_" + str(args.epoch) + ".pth")))


correct = 0
for (imgs, labels) in t_loader:
    imgs = Variable(imgs.cuda(args.gpu_id))
    pred = net(imgs)
    pred = F.softmax(pred)
    pred = pred.data.cpu().numpy()
    pred = pred.argmax(axis=1)
    labels = labels.numpy()
    correct += np.equal(labels, pred).sum()

correct = correct * 1.0 / len(t_set)
print "Epoch {0}: {1}".format(args.epoch, correct)
result.write("Epoch " + str(args.epoch) + ": " + str(correct) + "\n")
result.close()
