# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception

import io
import requests
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import pdb
import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument("--img_root")
parser.add_argument("--img_label",default="./data_label.txt")
parser.add_argument("--img_url")
parser.add_argument("--output_dir", default="./CAM_output")
parser.add_argument("--urlsave_dir", default="./url_photo")
parser.add_argument("--predict_root", default="./predict.txt")
args = parser.parse_args()


# input image
#LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
#IMG_URL = 'http://media.mlive.com/news_impact/photo/9933031-large.jpg'

# networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.
model_id = 2
if model_id == 1:
    net = models.squeezenet1_1(pretrained=True)
    finalconv_name = 'features' # this is the last conv layer of the network
elif model_id == 2:
    net = models.resnet18(pretrained=True)
    finalconv_name = 'layer4'
elif model_id == 3:
    net = models.densenet161(pretrained=True)
    finalconv_name = 'features'

net.eval()

# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

net._modules.get(finalconv_name).register_forward_hook(hook_feature)

# get the softmax weight
params = list(net.parameters())
weight_softmax = np.squeeze(params[-2].data.numpy())

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    _, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])

img_root = ""
if args.img_url is not None :
    response = requests.get(args.img_url)
    img_pil = Image.open(io.BytesIO(response.content))
    img_pil.save(os.path.join(args.urlsave_dir, args.img_url.split("/")[-1]))
    img_root = os.path.join(args.urlsave_dir, args.img_url.split("/")[-1])
elif args.img_root is not None:
    img_pil = Image.open(args.img_root)
    img_root = args.img_root
    #img_pil.save('test_img1.jpg')

img_tensor = preprocess(img_pil)
img_variable = Variable(img_tensor.unsqueeze(0))
logit = net(img_variable)

#mhk for prediction
data_label = args.img_label
fin = open(args.img_label, "r")
#label2predict = {}
id2predict = {}
count = 0
count1 = 0
for line in fin:
    data = line.strip()
    index = data.find(" ")
    #label2predict[data[:index]] = data[index+1:]
    id2predict[count1] = data[index+1:]
    count1 += 1

h_x = F.softmax(logit, dim=1).data.squeeze()
probs, idx = h_x.sort(0, True)
probs = probs.numpy()
idx = idx.numpy()

# output the prediction
for i in range(0, 5):
    print('{:.3f} -> {}'.format(probs[i], id2predict[idx[i]]))

# generate class activation mapping for the top1 prediction
CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

# render the CAM and output the predict
print('output CAM.jpg for the top1 prediction: %s'%id2predict[idx[0]])
fout = open(args.predict_root, "a")
jsondata = {}
jsondata["picture_name"] = img_root.split("/")[-1]
jsondata["confidence"] = str(probs[0])
jsondata["predict"] = str(id2predict[idx[0]])
fout.write(json.dumps(jsondata) + "\n")

# output the CAM.jpg
img = cv2.imread(img_root)
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img * 0.5
cv2.imwrite(os.path.join(args.output_dir, "CAM_" + img_root.split("/")[-1]), result)
