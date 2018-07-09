import torch
from torch.autograd import Variable
from torchvision import transforms as trn

import cv2
import numpy as np
from scipy.misc import imresize as imresize
from PIL import Image

import os

from modules.cnnvisualizer.utils import models

class Cnnvisualizer(object):

    # layer1 -  64 units
    # layer2 - 128 units
    # layer3 - 256 units
    # layer4 - 512 units

    def __init__(self, model_type='wideresnet_places365', threshold=0.2, image_size=None, segment_size=None):
        self.model_type = model_type
        self.threshold = threshold
        self.hook_handle = None

        if image_size is None:
            self.image_size = (224, 224)
        else:
            self.image_size = image_size

        if segment_size is None:
            self.segment_size = (120, 120)
        else:
            self.segment_size = segment_size 

        self.transform = trn.Compose([
                trn.Resize(self.image_size),
                trn.ToTensor(),
                trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_model(self):
        self.model = models.get_models(self.model_type)
        self.model.eval()
        self.model.cuda()

    def set_layer_unit(self, layer='layer4', unit=0):
        self._check_layer_unit_validation(layer, unit)
        
        self.layer = layer
        self.unit = unit

        self._register_feature_hook()

    def _check_layer_unit_validation(self, layer, unit):
        layer_max_map = {
            "layer1": 64,
            "layer2": 128,
            "layer3": 256,
            "layer4": 512
        }

        if (layer not in layer_max_map or
            unit < 0 or unit >= layer_max_map[layer]):
            raise ValueError(
                'Invalid layer and unit combination: '
                'max unit for each layer are {'
                'layer1: 63, layer2: 127, layer3: 255, layer4: 511}')

    def _register_feature_hook(self):
        self.feature_blob = None 
        if self.hook_handle is not None:
            self.hook_handle.remove()

        def hook_feature(module, input, output):
            self.feature_blob = np.squeeze(output.data.cpu().numpy())

        self.hook_handle = self.model._modules.get(self.layer).register_forward_hook(hook_feature)

    def read_input_image(self, input_path):
        return Image.open(input_path).convert('RGB')

    def generate_unitsegment(self, input_image):
        input = self._get_input_from_image(input_image)
        input = input.cuda()
        input = Variable(input, volatile=True)

        input_image_numpy = self._convert_image_to_numpy(input_image)
        logit = self.model.forward(input)

        segment = self._generate_unitsegment(input_image_numpy)

        return segment

    def _get_input_from_image(self, image):
        input = self.transform(image)
        input = input.unsqueeze(0)

        return input 

    def _convert_image_to_numpy(self, image):
        return np.array(image)

    def _generate_unitsegment(self, input_image_numpy):
        feature_map = self.feature_blob[self.unit]
        feature_map = self._normalize(feature_map)
    
        mask = self._get_mask(feature_map)
        img = self._resize_image(input_image_numpy, self.segment_size)
        segment = self._merge_image_mask(img, mask)

        return segment
    
    def _normalize(self, data):
        return data / (np.max(data) + self._get_eps())
    
    def _get_eps(self):
        return np.spacing(1)
    
    def _get_mask(self, feature_map):
        mask = self._resize_image(feature_map, self.segment_size)
    
        mask[mask < self.threshold] = 0.1
        mask[mask > self.threshold] = 1.0
    
        return mask
    
    def _resize_image(self, image, size):
        return cv2.resize(image, size)
    
    def _merge_image_mask(self, image, mask): 
        image = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        img_mask = np.multiply(image, mask[:, :, np.newaxis])
        img_mask = np.uint8(img_mask * 255)
    
        return img_mask

    def build_output_path(self, folder, prefix='default'):
        output_name = '%s_%s_%03d.jpg' % (prefix, self.layer, self.unit)
        return os.path.join(folder, output_name)

    def save_unitsegment(self, path, unitsegment):
        unitsegment = self._rgb2bgr(unitsegment)
        cv2.imwrite(path, unitsegment)

    def _rgb2bgr(self, img):
        (r, g, b) = cv2.split(img)
        return cv2.merge((b, g, r))

