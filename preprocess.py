"""
Data and preprocessing utils
"""

import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt

import numpy as np
from skimage.transform import pyramid_gaussian, pyramid_laplacian, resize

def get_pyramid(img, type, nlevels):
    pyramid = []
    pyramid_fn = pyramid_gaussian if type == 'gauss' else pyramid_laplacian
    for i in pyramid_fn(img, max_layer=nlevels-1, downscale=2, multichannel=True):
        pyramid.append(i)
    pyramid = [torch.FloatTensor(i.transpose(2,0,1)).unsqueeze(0) for i in pyramid]
    return pyramid

def preprocess_img(img, mask):
    img = resize(img, (128, 128))/255.0
    plt.show() 
    mask = 1 - resize(mask, (128, 128))/255.0
    mask[mask>0.1] = 1.0
    mask[mask<=0.1] = 0.0
    mask2 = np.dstack([mask, mask, mask])
    return img * mask2, mask