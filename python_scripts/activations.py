# -*- coding: utf-8 -*-
"""
This script is used to display the activations of the CNN for a given
input image.

Functions are from https://github.com/waleedka/cnn-visualization/blob/master/cnn_visualization.ipynb
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import skimage.io
import skimage.transform
import skimage.filters
from keras import backend as K
from keras.models import model_from_json
from scipy import misc

def tensor_summary(tensor):
    """Display shape, min, and max values of a tensor."""
    print("shape: {}  min: {}  max: {}".format(tensor.shape, tensor.min(), tensor.max()))
    return tensor.shape

    
def normalize(image):
    """Takes a tensor of 3 dimensions (height, width, colors) and normalizes it's values
    to be between 0 and 1 so it's suitable for displaying as an image."""
    image = image.astype(np.float32)
    return (image - image.min()) / (image.max() - image.min() + 1e-5)


def display_images(images, titles=None, size=5, interpolation=None, cmap="Greys_r"):
    """
    images: A list of images. I can be either:
        - A list of Numpy arrays. Each array represents an image.
        - A list of lists of Numpy arrays. In this case, the images in
          the inner lists are concatentated to make one image.
    """
    titles = titles or [""] * len(images)
#    rows = math.ceil(len(images) / cols)
#    height_ratio = 1.2 * (rows/cols) * (0.5 if type(images[0]) is not np.ndarray else 1)
    plt.figure(figsize=(size, size), dpi=120)
    gs1 = gridspec.GridSpec(size, size)
    gs1.update(wspace=1/size, hspace=1/size) # set the spacing between axes.
    i = 0
    for image, title in zip(images, titles):
        a = plt.subplot(gs1[i])
        a.autoscale_view('tight')
        plt.axis("off")
        # Is image a list? If so, merge them into one image.
        if type(image) is not np.ndarray:
            image = [normalize(g) for g in image]
            image = np.concatenate(image, axis=1)
        else:
            image = normalize(image)
        plt.title(title, fontsize=0)
        plt.imshow(image, cmap=cmap, interpolation=interpolation)
        i += 1
#    plt.savefig('act.png')
        
def read_layer(model, x, layer_name):
    """Returns the activation values for the specified layer"""
    # Create Keras function to read the output of a specific layer
    get_layer_output = K.function([model.layers[0].input], [model.get_layer(layer_name).output])
    outputs = get_layer_output([x])[0]
    num_filters = tensor_summary(outputs)[-1]
    return outputs[0], num_filters
    
def view_layer(model, x, layer_name, cols=5):
    """Returns a sum of all filter values for a specified layer (array)"""
    outputs = read_layer(model, x, layer_name)[0]
    num_filt = read_layer(model, x, layer_name)[1]
    size = int(np.ceil(np.sqrt(num_filt)))
    display_images([outputs[:,:,i] for i in range(num_filt)], size=size)
    heatmap = np.zeros(np.shape(outputs[:,:,0]))
    for i in range(num_filt):
        heatmap = heatmap + outputs[:,:,i]
    return heatmap
    
    
    
with tf.device('/cpu:0'):
    model_file = open('vgg19_branch.json', 'r')
    loaded_model = model_file.read()
    
    model = model_from_json(loaded_model)
    model.load_weights('vgg19_branch.h5')
    
    img = 'C:/Users/msawada/Desktop/trainottawa/45.445156,-75.671661/2012_QmOciqPiI4BhQbT0Ke8oXA.jpg'
    
    image = skimage.io.imread(img)
    x = image.astype(np.float32)
    x = [misc.imresize(x, (224,224))]
    
    heatmap = np.zeros((222,222))
    for i in range(1,17):
        heatmap = heatmap + misc.imresize(view_layer(model, x, "conv2d_%i"%i), (222,222), interp='bilinear')
    