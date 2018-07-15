
# coding: utf-8

# In[1]:

import matplotlib as mpl
mpl.use('Agg')

import numpy as np
import gzip, cPickle
from tsne import bh_sne

import os
import argparse
# In[2]:

parser = argparse.ArgumentParser()
parser.add_argument("--result_root", default="/home/shixun3/dyk/data")
parser.add_argument("--t", default="MNIST")
args = parser.parse_args()

output = np.load('train/output.npy').astype(np.float64)
data = np.load('train/data.npy')
target = np.load('train/target.npy')
print('data shape: ', data.shape)
print('target shape: ', target.shape)
print('output shape: ', output.shape)


# In[3]:

output_2d = bh_sne(output)


# In[4]:


np.save('train/output_2d.npy', output_2d, allow_pickle=False)


# In[5]:

import matplotlib
import matplotlib.pyplot as plt


# In[6]:
colors = ["red", "cyan", "gold", "yellow", "green", "orange", "purple", "silver", "blue", "black"]

plt.rcParams['figure.figsize'] = 20, 20
plt.scatter(output_2d[:, 0], output_2d[:, 1], c=target*1000)
plt.savefig(os.path.join(args.result_root, args.t, "tsne.png"), bbox_inches='tight')




