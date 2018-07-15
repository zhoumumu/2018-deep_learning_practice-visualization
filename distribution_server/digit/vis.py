import matplotlib as mpl
mpl.use('Agg')

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable

from utils import DigitImage
from model import resnet64

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="target")
parser.add_argument("--result_root", default="/home/shixun3/dyk/data")
parser.add_argument("--data_root", default="/home/shixun3/dyk/data")
parser.add_argument("--s", default="SVHN")
parser.add_argument("--t", default="MNIST")
parser.add_argument("--batch_size", default=256)
parser.add_argument("--shuffle", default=False)
parser.add_argument("--num_workers", default=8)
parser.add_argument("--snapshot", default="/home/shixun3/dyk/digit/snapshot")
parser.add_argument("--epoch", default=40)
parser.add_argument("--gpu_id", default=0)
parser.add_argument("--vis_dims", default=2)
parser.add_argument("--class_num", default=10)
parser.add_argument("--extract", default=True)
args = parser.parse_args()

s_root = os.path.join(args.data_root, args.s, "imgs")
s_label = os.path.join(args.data_root, args.s, "label.txt")
t_root = os.path.join(args.data_root, args.t, "imgs")
t_label = os.path.join(args.data_root, args.t, "label.txt")
data_transform = transforms.Compose([
    transforms.Scale((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
s_set = DigitImage(s_root, s_label, data_transform)
t_set = DigitImage(t_root, t_label, data_transform)
#assert len(s_set) == 9000
#assert len(t_set) == 9000
s_loader = torch.utils.data.DataLoader(s_set, batch_size=args.batch_size,
    shuffle=args.shuffle, num_workers=args.num_workers)
t_loader = torch.utils.data.DataLoader(t_set, batch_size=args.batch_size,
    shuffle=args.shuffle, num_workers=args.num_workers)

net = resnet64(vis_dims=args.vis_dims, class_num=args.class_num, extract=args.extract).cuda(args.gpu_id)
net.eval()
net.load_state_dict(torch.load(os.path.join(args.snapshot, "SVHN_Source_only_resnet64_" + str(args.epoch) + ".pth")))

#cm = pylab.get_cmap('gist_rainbow')
def plot_embedding(source_X, source_Y, source_gt, target_X, target_Y, target_gt, title=None):
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    colors = ["red", "cyan", "gold", "yellow", "green",
             "orange", "purple", "silver", "blue", "black"]
    plt.figure(figsize=(15, 15))
    #plt.scatter(X, Y, color=colors)
    """
    normalize = mcolors.Normalize(vmin=0, vmax=args.class_num-1)
    cmap = cm.rainbow
    plt.figure(figsize=(15, 15))
    plt.scatter(X, Y, color=cmap(normalize(args.class_num)))
    scalarmappable = cm.ScalarMappable(norm=normalize, cmap=cmap)
    scalarmappable.set_array([])
    plt.colorbar(scalarmappable)
    """
    
    for i in range(len(source_X)):
        plt.scatter(source_X[i], source_Y[i], marker=">", color=colors[source_gt[i]])

    for i in range(len(target_X)):
        plt.scatter(target_X[i], target_Y[i], marker="+", color=colors[target_gt[i]])
    
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.savefig(os.path.join(args.result_root, "embedding.png"), bbox_inches='tight')


def plot_target(target_X, target_Y, target_gt, title=None):
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    colors = ["red", "cyan", "gold", "yellow", "green",
             "orange", "purple", "silver", "blue", "black"]
    plt.figure(figsize=(15, 15))
    #plt.scatter(X, Y, color=colors)
    """
    normalize = mcolors.Normalize(vmin=0, vmax=args.class_num-1)
    cmap = cm.rainbow
    plt.figure(figsize=(15, 15))
    plt.scatter(X, Y, color=cmap(normalize(args.class_num)))
    scalarmappable = cm.ScalarMappable(norm=normalize, cmap=cmap)
    scalarmappable.set_array([])
    plt.colorbar(scalarmappable)
    """
    for i in range(len(target_X)):
        plt.scatter(target_X[i], target_Y[i], marker="+", color=colors[target_gt[i]])
    
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.savefig(os.path.join(args.result_root, args.t, "digit.png"), bbox_inches='tight')


source_X = []
source_Y = []
source_gt = []
for (imgs, labels) in s_loader:
    imgs = Variable(imgs.cuda(args.gpu_id))
    embedding = net(imgs)
    embedding = embedding.data.cpu().numpy()
    labels = labels.numpy()
    for i in range(embedding.shape[0]):
        source_X.append(embedding[i][0])
        source_Y.append(embedding[i][1])
        source_gt.append(labels[i])

target_X = []
target_Y = []
target_gt = []
for (imgs, labels) in t_loader:
    imgs = Variable(imgs.cuda(args.gpu_id))
    embedding = net(imgs)
    embedding = embedding.data.cpu().numpy()
    labels = labels.numpy()
    for i in range(embedding.shape[0]):
        target_X.append(embedding[i][0])
        target_Y.append(embedding[i][1])
        target_gt.append(labels[i])

if args.mode=="embedding":
    plot_embedding(source_X, source_Y, source_gt, target_X, target_Y, target_gt, "Embedding")
elif args.mode=="target":
    plot_target(target_X, target_Y, target_gt)
