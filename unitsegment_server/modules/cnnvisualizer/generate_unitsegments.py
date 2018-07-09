import torch
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data as data

import torchvision.models as models
from torchvision import transforms as trn

import cv2
import numpy as np
from scipy.misc import imresize as imresize
from PIL import Image

import os
import argparse

from data.dataset import Dataset
from utils import models

parser = argparse.ArgumentParser()
parser.add_argument('--input_list', type=str, default='datasets/images/sample.txt')
parser.add_argument('--output_folder', type=str, default='results/')
parser.add_argument('--output_prefix', type=str, default='default')

parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--segment_size', type=int, default=120)
parser.add_argument('--threshold', type=float, default=0.2, help='Smaller the segmentation will be tighter.')
parser.add_argument('--top', type=int, default=12, help='Number of top units')

parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--worker', type=int, default=6, help='Number of workers')

parser.add_argument('--model', type=str, default='wideresnet_places365')
parser.add_argument('--layer', type=str, default='layer4', help='split by ,')
opt = parser.parse_args() 

def main():
    img_size = (opt.image_size, opt.image_size)
    segment_size = (opt.segment_size, opt.segment_size)
    features_names = opt.layer.split(',')
    
    model = models.get_models(opt.model)
    model.eval()
    model.cuda()
    # print(model)
    features_blobs = register_hook(model, features_names)
    
    input_image_list = get_input_image_list(opt.input_list)
    loader = get_data_loader(input_image_list, img_size, opt.batch_size, opt.worker)

    max_features = extract_max_features(model, loader, features_names, features_blobs)

    for layer_id, layer in enumerate(features_names):
        num_units = max_features[layer_id].shape[1]
        print('max_features: ', max_features[layer_id].shape)
        image_list_sorted = []

        for unit_id in range(num_units):
            # img x unit
            activations_unit = np.squeeze(max_features[layer_id][:, unit_id])
            idx_sorted = np.argsort(activations_unit)[::-1]
            if unit_id == 0:
                print('activations_unit: ', activations_unit.shape)
                print('idx_sorted: ', idx_sorted)
            image_list_sorted += [input_image_list[item] for item in idx_sorted[:opt.top]]
        print('image_list_sorted len:', len(image_list_sorted))

        loader_top = get_data_loader(
                image_list=image_list_sorted,
                image_size=img_size,
                batch_size=opt.top,
                num_workers=opt.worker)

        for unit_id, (input, paths) in enumerate(loader_top):
            del features_blobs[:]
            
            print('[%s] %d / %d' % (layer, unit_id, num_units))
            input = input.cuda()
            input_var = Variable(input, volatile=True)
            logit = model.forward(input_var)

            # num_top x units_num x height x width
            feature_maps = features_blobs[layer_id]
            
            segments = generate_layer_top_units_segments(unit_id, feature_maps, paths, segment_size, opt.threshold)
            output_paths = generate_output_paths(segments, opt.output_folder, opt.output_prefix, layer, unit_id)
            save_segments(segments, output_paths)


def register_hook(model, features):
    features_blobs = []

    def hook_feature(module, input, output):
        features_blobs.append(np.squeeze(output.data.cpu().numpy()))

    for name in features:
        model._modules.get(name).register_forward_hook(hook_feature)

    return features_blobs

def get_input_image_list(input_list_path):
    with open(input_list_path, 'r') as f:
        img_list = [line.rstrip() for line in f]

    return img_list

def get_data_loader(image_list, image_size, batch_size, num_workers, shuffle=False):
    tf = trn.Compose([
            trn.Resize(image_size),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = Dataset(image_list, tf)

    data_loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle
    )

    return data_loader

# @return
# length: features_num
# element size: dataset_size x unit_num
def extract_max_features(model, loader, features_names, features_blobs):
    batch_size = loader.batch_size
    dataset = loader.dataset

    max_features = [None for feature in features_names]
    
    for batch_idx, (input, paths) in enumerate(loader):
        del features_blobs[:]

        print('[extract_max_features] %d / %d' % (batch_idx + 1, len(loader)))

        input = input.cuda()
        input = Variable(input, volatile=True)
        logit = model.forward(input)

        if max_features[0] is None:
            for idx, feat_batch in enumerate(features_blobs):
                feature_size = (len(dataset), feat_batch.shape[1])
                print(idx, feature_size)
                max_features[idx] = np.zeros(feature_size)

        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))

        for idx, feat_batch in enumerate(features_blobs):
            # feat_batch.shape = batch_size x unit_num x height x width
            max_features[idx][start_idx:end_idx] = np.max(np.max(feat_batch, 3), 2)  # batch_size x unit_num

    return max_features

def generate_layer_top_units_segments(unit_id, feature_maps, input_paths, segment_size, threshold):
    num_top = feature_maps.shape[0]
    num_units = feature_maps.shape[1]
    segment_list = []

    for i in range(num_top):
        input_image = read_image(input_paths[i])
        segment = generate_single_unit_segments(feature_maps[i][unit_id], input_image, segment_size, threshold)
        segment_list.append(segment)

    return segment_list

def generate_single_unit_segments(feature_map, input_image, segment_size, threshold):
    feature_map = normalize(feature_map)

    mask = get_mask(feature_map, segment_size, threshold)
    img = resize_image(input_image, segment_size)
    segment = merge_image_mask(img, mask)

    return segment

def normalize(data):
    return data / (np.max(data) + get_eps())

def get_eps():
    return np.spacing(1)

def get_mask(feature_map, segment_size, threshold):
    mask = resize_image(feature_map, segment_size)

    mask[mask < threshold] = 0.2
    mask[mask > threshold] = 1.0

    return mask

def read_image(path):
    return cv2.imread(path)

def resize_image(image, size):
    return cv2.resize(image, size)

def merge_image_mask(image, mask): 
    image = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    img_mask = np.multiply(image, mask[:, :, np.newaxis])
    img_mask = np.uint8(img_mask * 255)

    return img_mask

def generate_output_paths(segments, output_folder, output_prefix, layer, unit):
    paths = []

    for top in range(len(segments)):
        name = '%s_%s_%d_%d.jpg' % (output_prefix, layer, unit, top) 
        paths.append(os.path.join(output_folder, name))

    return paths

def save_segments(segments, output_paths):
    for path, segment in zip(output_paths, segments):
        save_image(path, segment)

def save_image(path, image):
    cv2.imwrite(path, image)

if __name__ == '__main__':
    main()
