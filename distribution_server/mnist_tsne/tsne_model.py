# Demo
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
import os
import numpy as np
from tsne import bh_sne
import matplotlib.pyplot as plt
from PIL import Image

from utils import DigitImage

class TsneModel(object):
    def __init__(self, no_cuda=False, save_dir='./train', seed=1, data_root="/home/shixun3/dyk/data", batch_size=64,
                 t="MNIST", result_root='/home/shixun3/dyk/data'):
        self.no_cuda = no_cuda
        self.save_dir = save_dir
        self.seed = seed
        self.data_root = data_root
        self.batch_size = batch_size
        self.t = t
        self.result_root = result_root

        self.cuda = not self.no_cuda and torch.cuda.is_available()

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        torch.manual_seed(self.seed)
        if self.cuda:
            torch.cuda.manual_seed(self.seed)

        self.t_root = os.path.join(self.data_root, self.t, "imgs")
        self.t_label = os.path.join(self.data_root, self.t, "label.txt")

        t_set = DigitImage(self.t_root, self.t_label, transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))

        self.kwargs = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}

        self.train_loader = torch.utils.data.DataLoader(t_set, batch_size=self.batch_size, shuffle=True, **self.kwargs)

        # load model
        self.model = Net()
        self.model.eval()
        model_path = os.path.join(self.save_dir, 'mnist_bn.pth')
        self.model.load_state_dict(torch.load(model_path))


    def api_function(self, *params):
        if params[0]:
            self.t = params[0]

        self.t_root = os.path.join(self.data_root, self.t, "imgs")
        self.t_label = os.path.join(self.data_root, self.t, "label.txt")
        t_set = DigitImage(self.t_root, self.t_label, transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
        self.train_loader = torch.utils.data.DataLoader(t_set, batch_size=self.batch_size, shuffle=True, **self.kwargs)

        self.generate_feature()

        output = np.load('train/output.npy').astype(np.float64)
        data = np.load('train/data.npy')
        target = np.load('train/target.npy')
        print('data shape: ', data.shape)
        print('target shape: ', target.shape)
        print('output shape: ', output.shape)

        output_2d = bh_sne(output)
        np.save('train/output_2d.npy', output_2d, allow_pickle=False)

        plt.rcParams['figure.figsize'] = 20, 20
        plt.scatter(output_2d[:, 0], output_2d[:, 1], c=target * 10)
        plt.savefig(os.path.join(self.result_root, self.t, "tsne.png"), bbox_inches='tight')

        return dict(
            data=Image.open(os.path.join(self.result_root, self.t, "tsne.png"))
        )


    def generate_feature(self, model):
        cnt = 0
        out_target = []
        out_data = []
        out_output = []
        for data, target in self.train_loader:
            cnt += len(data)
            print("processing: %d/%d" % (cnt, len(self.train_loader.dataset)))
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            output_np = output.data.cpu().numpy()
            target_np = target.data.cpu().numpy()
            data_np = data.data.cpu().numpy()

            out_output.append(output_np)
            out_target.append(target_np[:, np.newaxis])
            out_data.append(np.squeeze(data_np))

        output_array = np.concatenate(out_output, axis=0)
        target_array = np.concatenate(out_target, axis=0)
        data_array = np.concatenate(out_data, axis=0)

        print(output_array.shape)
        print(target_array.shape)
        print(data_array.shape)

        np.save(os.path.join(self.save_dir, 'output.npy'), output_array, allow_pickle=False)
        np.save(os.path.join(self.save_dir, 'target.npy'), target_array, allow_pickle=False)
        np.save(os.path.join(self.save_dir, 'data.npy'), data_array, allow_pickle=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv1_bn = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_bn = nn.BatchNorm2d(20)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc1_bn = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        bn_1 = self.conv1_bn(self.conv1(x))
        x = F.relu(F.max_pool2d(bn_1, 2))
        bn_2 = self.conv2_bn(self.conv2(x))
        x = F.relu(F.max_pool2d(self.conv2_drop(bn_2), 2))
        x = x.view(-1, 320)
        bn_fc = self.fc1_bn(self.fc1(x))
        x = F.relu(bn_fc)
        x = F.dropout(x, training=self.training)
        return x


tsne_model_instance = TsneModel()
