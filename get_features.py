import argparse
import os
import sys
import torch
import torch.nn as nn

import torchvision.models as models
import models.vgg16 as VGG16
import models.vgg19 as VGG19

def MakeFeatureModel(modelName='vgg16'):
    """Returns a perceptual loss network that outputs feature maps"""
    # Construct the feature extraction model
    models = ['vgg16','vgg19']
    if modelName not in models:
        raise ValueError('Invalid model name; available models: {}'.format(models))

    elif modelName == models[0]: #vgg16 pretrained on imagenet
        FeatureModel = VGG16.vgg16(pretrained=True ,feat_ex=True)
    elif modelName == models[1]: #vgg19 pretrained on imagenet
        FeatureModel = VGG19.vgg19(pretrained=True ,feat_ex=True)

    return FeatureModel

def GetFeatures(images, FeatureModel, UseCuda=True):
    """ Gets feature maps given input image tensors of dimensions (Batch,C,H,W).
    
    This function expects input image pixel values to be scaled between 0 to 1.
    """
    assert type(images) == type(torch.tensor(0)), 'Input images must be a torch tensor of dimensions (Batch,C,H,W)'
    InputDevice = images.device
    if UseCuda:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            raise Warning('UseCuda was set true but cuda is not available, using cpu')
    else:
        device = 'cpu'

    FeatureModel = FeatureModel.to(device)
    images = images.to(device)

    # Disable gradients in model
    for param in FeatureModel.parameters():
        param.requires_grad = False

    # Standardise the input for PyTorch pretrained models
    # Perform clone to avoid inplace operation disrupting gradient
    imagesNorm = images.clone()
    # Subtract means
    imagesNorm[:,0,:,:] -= 0.485
    imagesNorm[:,1,:,:] -= 0.456
    imagesNorm[:,2,:,:] -= 0.406
    # Divide by SDs
    imagesNorm[:,0,:,:] /= 0.229
    imagesNorm[:,1,:,:] /= 0.224
    imagesNorm[:,2,:,:] /= 0.225

    # Get the features
    Features = FeatureModel(imagesNorm)
    
    # Send to device of original input
    Features =  [Feature.to(InputDevice) for Feature in Features]
    return Features