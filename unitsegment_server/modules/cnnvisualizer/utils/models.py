import torch
import torchvision.models as models 

import os
from functools import partial
import pickle

pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

basedir = os.path.abspath(os.path.dirname(__file__))
rootdir = os.path.abspath(os.path.dirname(basedir))

snapshots_base = os.path.join(rootdir, 'snapshots')
model_list = [
    'wideresnet_places365',
    'resnet18',
    'squeezenet']

def get_models(model_name):
    if model_name == 'wideresnet_places365':
        return get_wideresnet()
    elif model_name == 'resnet18':
        return get_resnet18()
    elif model_name == 'squeezenet':
        return get_squeezenet()
    else:
        raise ValueError('Invalid model name')

def get_wideresnet():
    model_file = 'whole_wideresnet18_places365.pth.tar'
    model_path = get_wideresnet_path(model_file)

    return load_model(model_path)

def get_wideresnet_path(model_file):
    return os.path.join(snapshots_base, 'wideresnet18', model_file)

def load_model(model_path):
    print('load_model:', model_path)
    model = torch.load(model_path, map_location=lambda storage, loc: storage, pickle_module=pickle)
    return model 
    
def get_resnet18():
    return models.resnet18(pretrained=True)

def get_squeezenet():
    return models.squeezenet1_0(pretrained=True)

