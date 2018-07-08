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

img_label = r"./data_label.txt"
output_dir = r"./CAM_output"
urlsave_dir = r"./realimg"
predict_root = r"./predict.txt"

count = 1

# hook the feature extractor
features_blobs = []

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

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

def id2predictlabel(img_label):
    #data_label = img_label
    fin = open(img_label, "r")
    #label2predict = {}
    id2predict = {}
    count1 = 0
    for line in fin:
        data = line.strip()
        index = data.find(" ")
        #label2predict[data[:index]] = data[index+1:]
        id2predict[count1] = data[index+1:]
        count1 += 1
    return id2predict

def output_predict(probs, idx, id2predict):
    for i in range(0, 5):
        print('{:.3f} -> {}'.format(probs[i], id2predict[idx[i]]))

def renderCAM(idx, id2predict, img_root, probs):
    #print('output CAM.jpg for the top1 prediction: %s'%id2predict[idx[0]])
    predict_root = "./predict.txt"
    fout = open(predict_root, "a")
    jsondata = {}
    jsondata["picture_name"] = img_root.split("/")[-1]
    jsondata["confidence"] = str(probs[0])
    jsondata["predict"] = str(id2predict[idx[0]])
    fout.write(json.dumps(jsondata, ensure_ascii=False) + "\n")
    return json.dumps(jsondata, ensure_ascii=False)

def outputCAM(img_root, CAMs, output_dir):
    img = cv2.imread(img_root)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(
        CAMs[0], (width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite(os.path.join(output_dir, "CAM_" +
                            img_root.split("/")[-1]), result)
    campath = os.path.join(output_dir, "CAM_" + img_root.split("/")[-1])
    return campath

def returnpredict(img_root, flag):
    net = models.resnet18(pretrained=True)
    finalconv_name = 'layer4'
    net.eval()
    net._modules.get(finalconv_name).register_forward_hook(hook_feature)
    # get the softmax weight
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    if flag == 0:
        global count
        response = requests.get(img_root)
        img_pil = Image.open(io.BytesIO(response.content))
        img_pil.save(os.path.join(urlsave_dir, str(count) + ".jpg"))
        img_root = os.path.join(urlsave_dir, str(count) + ".jpg")        
        count += 1
    else:
        img_pil = Image.open(img_root)
        img_root = img_root
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    logit = net(img_variable)
    id2predict = id2predictlabel(img_label)
    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()
    #output_predict(probs, idx, id2predict)
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
    predict = renderCAM(idx, id2predict, img_root, probs)
    campath = outputCAM(img_root, CAMs, output_dir)
    return campath, predict, idx[0], img_root
