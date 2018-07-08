"""
Created on Fri Dec 15 19:57:34 2017
@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import numpy as np
import cv2

import torch
from torch import nn
from torch.autograd import Variable
# from torch.autograd.gradcheck import zero_gradients  # See processed_image.grad = None

from misc_functions import preprocess_image, recreate_image, get_params
from torchvision import models

class FastGradientSignUntargeted():
    """
        Fast gradient sign untargeted adversarial attack, minimizes the initial class activation
        with iterative grad sign updates
    """
    def __init__(self, model, alpha):
        self.model = model
        self.model.eval()
        # Movement multiplier per iteration
        self.alpha = alpha
        # Create the folder to export images if not exists
        if not os.path.exists('../generated'):
            os.makedirs('../generated')

    def generate(self, original_image, im_label, image_name):
        # I honestly dont know a better way to create a variable with specific value
        height, width, _ = original_image.shape
        #print(height, width)
        gampicpath = ""
        im_label_as_var = Variable(torch.from_numpy(np.asarray([im_label])))
        # Define loss functions
        ce_loss = nn.CrossEntropyLoss()

        # Resize original_image
        re_or_image = cv2.resize(original_image, (224, 224))
        re_or_image = np.float32(re_or_image)
        re_or_image = np.ascontiguousarray(re_or_image[..., ::-1])
        
        
        # Process image
        processed_image = preprocess_image(original_image)
        # Start iteration
        for i in range(10):
            #print('Iteration:', str(i))
            # zero_gradients(x)
            # Zero out previous gradients
            # Can also use zero_gradients(x)
            processed_image.grad = None
            # Forward pass
            out = self.model(processed_image)
            # Calculate CE loss
            pred_loss = ce_loss(out, im_label_as_var)
            # Do backward pass
            pred_loss.backward()
            # Create Noise
            # Here, processed_image.grad.data is also the same thing is the backward gradient from
            # the first layer, can use that with hooks as well
            adv_noise = self.alpha * torch.sign(processed_image.grad.data)
            # Add Noise to processed image
            processed_image.data = processed_image.data + adv_noise

            # Confirming if the image is indeed adversarial with added noise
            # This is necessary (for some cases) because when we recreate image
            # the values become integers between 1 and 255 and sometimes the adversariality
            # is lost in the recreation process

            # Generate confirmation image
            recreated_image = recreate_image(processed_image)
            # Process confirmation image
            prep_confirmation_image = preprocess_image(recreated_image)
            # Forward pass
            confirmation_out = self.model(prep_confirmation_image)
            # Get prediction
            _, confirmation_prediction = confirmation_out.data.max(1)
            # Get Probability
            confirmation_confidence = \
                nn.functional.softmax(confirmation_out)[0][confirmation_prediction].data.numpy()[0]
            # Convert tensor to int
            confirmation_prediction = confirmation_prediction.numpy()[0]
            # Check if the prediction is different than the original
            if (i < 9) & (confirmation_confidence < 0.5):
                continue
            if confirmation_prediction != im_label:
                # print('Original image was predicted as:', im_label,
                #       'with adversarial noise converted to:', confirmation_prediction,
                #       'and predicted with confidence of:', confirmation_confidence)
                # Create the image for noise as: Original image - generated image
                # noise_image = re_or_image - recreated_image
                # cv2.imwrite('../generated/untargeted_adv_noise_from_' + str(im_label) + '_to_' +
                #             str(confirmation_prediction) + '.jpg', noise_image)
                # Write image
                print(type(recreated_image))
                out_img = cv2.resize(recreated_image, (width, height))             
                cv2.imwrite('./generated/adv_' + image_name, out_img)
                gampicpath = './generated/adv_' + image_name
                break

        return gampicpath


def generate_un_ad_sample(img_path, img_label):
    """
    Args Type: str, int

    Result path: '../generated/adv_image_name'
    """

    pretrained_model = models.resnet18(pretrained=True)
    FGS_untargeted = FastGradientSignUntargeted(pretrained_model, 0.01)
    (original_image, prep_img, target_class) = get_params(img_path, img_label)

    img_name = img_path.split('/')[-1]
    gampicpath = FGS_untargeted.generate(original_image, target_class, img_name)
    return gampicpath
