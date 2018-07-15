# Demo
from __future__ import print_function
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from utils import DigitImage
from model import resnet64

class DigitModel(object):
    def __init__(self, save_dir='./train', data_root="/home/shixun3/dyk/data", batch_size=256, shuffle=False,
                 t="MNIST", s="SVHN", result_root='/home/shixun3/dyk/data', num_workers=8, vis_dims=2, gpu_id=0,
                 extract=True, class_num=10, snapshot="/home/shixun3/dyk/digit/snapshot", epoch=40):
        self.save_dir = save_dir
        self.data_root = data_root
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.t = t
        self.s = s
        self.num_workers = num_workers
        self.result_root = result_root
        self.vis_dims = vis_dims
        self.gpu_id = gpu_id
        self.class_num = class_num
        self.extract = extract
        self.snapshot = snapshot
        self.epoch = epoch

        self.s_root = os.path.join(self.data_root, self.s, "imgs")
        self.s_label = os.path.join(self.data_root, self.s, "label.txt")
        self.t_root = os.path.join(self.data_root, self.t, "imgs")
        self.t_label = os.path.join(self.data_root, self.t, "label.txt")
        data_transform = transforms.Compose([
            transforms.Scale((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.s_set = DigitImage(self.s_root, self.s_label, data_transform)
        self.t_set = DigitImage(self.t_root, self.t_label, data_transform)

        self.s_loader = torch.utils.data.DataLoader(self.s_set, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        self.t_loader = torch.utils.data.DataLoader(self.t_set, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

        self.model = resnet64(vis_dims=self.vis_dims, class_num=self.class_num, extract=self.extract).cuda(self.gpu_id)
        self.model.eval()
        self.model.load_state_dict(torch.load(os.path.join(self.snapshot, "SVHN_Source_only_resnet64_" + str(self.epoch) + ".pth")))

        source_X = []
        source_Y = []
        source_gt = []
        for (imgs, labels) in self.s_loader:
            imgs = Variable(imgs.cuda(self.gpu_id))
            embedding = self.model(imgs)
            embedding = embedding.data.cpu().numpy()
            labels = labels.numpy()
            for i in range(embedding.shape[0]):
                source_X.append(embedding[i][0])
                source_Y.append(embedding[i][1])
                source_gt.append(labels[i])

        self.target_X = []
        self.target_Y = []
        self.target_gt = []
        for (imgs, labels) in self.t_loader:
            imgs = Variable(imgs.cuda(self.gpu_id))
            embedding = self.model(imgs)
            embedding = embedding.data.cpu().numpy()
            labels = labels.numpy()
            for i in range(embedding.shape[0]):
                self.target_X.append(embedding[i][0])
                self.target_Y.append(embedding[i][1])
                self.target_gt.append(labels[i])



    def api_function(self, *params):
        if params[0]:
            self.t = params[0]

        self.t_root = os.path.join(self.data_root, self.t, "imgs")
        self.t_label = os.path.join(self.data_root, self.t, "label.txt")
        data_transform = transforms.Compose([
            transforms.Scale((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.t_set = DigitImage(self.t_root, self.t_label, data_transform)
        self.t_loader = torch.utils.data.DataLoader(self.t_set, batch_size=self.batch_size,
                                                    shuffle=self.shuffle, num_workers=self.num_workers)

        self.plot_target(self.target_X, self.target_Y, self.target_gt)

        return dict(
            data=Image.open(os.path.join(self.result_root, self.t, "digit.png"))
        )

    def plot_target(self, target_X, target_Y, target_gt, title=None):
        colors = ["red", "cyan", "gold", "yellow", "green",
                  "orange", "purple", "silver", "blue", "black"]
        plt.figure(figsize=(15, 15))
        # plt.scatter(X, Y, color=colors)
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
        plt.savefig(os.path.join(self.result_root, self.t, "digit.png"), bbox_inches='tight')


digit_model_instance = DigitModel()
