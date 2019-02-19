from __future__ import print_function
# coding: utf-8

# be sure to be in environment: **Environment (conda_tensorflow_p27)**

# In[38]:

import scipy.misc
import tensorflow as tf
import numpy as np
import time
from IPython.display import Image
from PIL import Image

from keras.preprocessing.image import load_img, img_to_array
from scipy.misc import imsave # to be able to save as an image
from scipy.optimize import fmin_l_bfgs_b # to optimize our cost function
import imageio

from keras.applications import vgg19 #to load vgg19 network
from keras import backend as K
import os

import keras
tf.__version__
keras.__version__

from keras.utils.vis_utils import plot_model #to be able to visualize the network
# get_ipython().magic('matplotlib inline')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


# ## Config and Hyperparameters for Model Tunning

# In[61]:

'''
# change below to edit images and model params
'''
size = 2                                             # n images generate
trial_name = 'fun'                                   # give this trial a name so all output files can be stored
content_file_name = '.png'                           # content image
style_file_name = 'vuillard_1.jpg'                   # style image
iterations = 13                                      # epochs during style fitting
total_variation_weight = 1.0                         # 
style_weight = 5                                     # weight assinged to style 
content_weight = 0.025                               # weight assinged to content
content_features = ['block5_conv2']                  # feature for content image
style_features = ['block1_conv1',                    # features of the style image
                  'block2_conv1',
                  'block3_conv1', 
                  'block4_conv1',
                  'block5_conv1']


# run script over a lot of trials
config = {
    'cfn'    : ['jing_noise2.png','young_noise.png'],
    'sfn'    : ['monet2.jpg','vuillard_1.jpg'],
    'sw'     : [5,5]
}

def get_rand_cfeatures():
    d = {}
    i = 0
    
    while i < size:
        block = np.random.choice([1,2,3,4,5])
        if block <= 2:
            convo = np.random.choice([1,2])
        else:
            convo = np.random.choice([1,2,3,4])
        
        d[i] = ['block' + str(block) + '_' + 'conv' + str(convo)]
        i += 1
    
    return d

def get_rand_ffeatures():
    d = {}
    i = 0

    while i < size:
        ls = []
        j = 0
        while j < 5:
            block = np.random.choice([1,2,3,4,5])
            if block <= 2:
                convo = np.random.choice([1,2])
            else:
                convo = np.random.choice([1,2,3,4])
            
            string = 'block' + str(block) + '_' + 'conv' + str(convo)
            ls.append(string)
            j +=1
        d[i] = ls
        i += 1
    
    return d

config_cfeatures = get_rand_cfeatures() 
config_ffeatures = get_rand_ffeatures()


#%%
w = 0 
while w < size:
    print(w)
    content_file_name = config['cfn'][w]
    style_file_name = config['sfn'][w]
    style_weight = config['sw'][w]
    content_features = config_cfeatures[w]
    style_features = config_ffeatures[w]
    print(content_features)
    print(style_features)
    print(style_weight)
    trial_name = 'rand_'
    trial_name = trial_name + content_file_name[:-4] + '_' + style_file_name[:-4] + '_' + str(style_weight)
    print(trial_name)

#%%
    '''
    # config definitions... do not change
    '''
    base_image_path = os.getcwd() + '/' + content_file_name
    style_reference_image_path = os.getcwd() + '/' + style_file_name
    img_nrows = 500  # scale the image to n pixel rows 
    width, height = load_img(base_image_path).size 
    img_ncols = int(width * img_nrows / height) # scale cols pixels


    # ## Preview Style and Content Images

    # In[62]:

    def preview_imgs():
        fontsize = 15
        fig, (ax1,ax2) = plt.subplots(2,1,figsize = (8,8))

        # image one 
        ax1.set_title('Content Image',fontsize=fontsize)
        ax1.imshow(load_img(base_image_path))
        ax1.set_yticks([])
        ax1.set_xticks([])

        # image 2
        ax2.set_title('Style Image',fontsize=fontsize)
        ax2.set_yticks([])
        ax2.set_xticks([])
        ax2.imshow(load_img(style_reference_image_path))
        fig

    plt.close('all')
    preview_imgs()


    # ## Image Preprocssing

    # In[63]:

    '''
    will rescale a image based based off the global definition of n_row and n_cols
    and convert that image into a 3-D tensor for model purposes
    '''
    def preprocess_image(image_path):
        width, height = load_img(base_image_path).size
        img_ncols = int(width * img_nrows / height)
        img = load_img(image_path, target_size=(img_nrows, img_ncols))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = vgg19.preprocess_input(img)
        return img


    # will convert a tensor into a valid image
    def deprocess_image(x):
        if K.image_data_format() == 'channels_first':
            x = x.reshape((3, img_nrows, img_ncols))
            x = x.transpose((1, 2, 0))
        else:
            x = x.reshape((img_nrows, img_ncols, 3))

        # Remove zero-center by mean pixel
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68

        # 'BGR'->'RGB'

        x = x[:, :, ::-1]
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    # will init the output page of style transfer to content
    def init_combination_image():
        # this will contain our generated image
        if K.image_data_format() == 'channels_first':
            return K.placeholder((1, 3, img_nrows, img_ncols))
        else:
            return K.placeholder((1, img_nrows, img_ncols, 3))


        

    base_image = K.variable(preprocess_image(base_image_path))
    style_reference_image = K.variable(preprocess_image(style_reference_image_path))
    combination_image = init_combination_image()

    # combine the 3 images (style, content, result image that starts from the white noise) into a single Keras tensor
    input_tensor = K.concatenate([base_image,
                                style_reference_image,
                                combination_image], axis=0)


    # ## Generate Model

    # In[64]:

    # build the VGG19 network with our 3 images as input
    # the model will be loaded with pre-trained ImageNet weights
    model = vgg19.VGG19(input_tensor=input_tensor,weights='imagenet', include_top=False)

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    def write_layers(dct):
        fn = open(os.getcwd() + '/vgg19_layers.txt','w')

        for i in list(dct):
            fn.write(i)
            fn.write('\t')
            fn.write(str(dct[i]))
            fn.write('\n')
        fn.close()
    write_layers(outputs_dict)
                                    
    for i in style_features:
        assert i in list(outputs_dict)

    for j in content_features:
        print('dlkfjdlk')
        print(j)
        print('sdlkljsf;')
        print(list(outputs_dict))
        assert j in list(outputs_dict)


    # ## Loss Functions

    # In[65]:

    # compute the neural style loss
    # first we need to define 4 utility functions
    # the gram matrix of an image tensor (feature-wise outer product)

    def gram_matrix(x):

        assert K.ndim(x) == 3
        if K.image_data_format() == 'channels_first':
            features = K.batch_flatten(x)
        else:
            features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))

        gram = K.dot(features, K.transpose(features))
        return gram

    # the "style loss" is designed to maintain
    # the style of the reference image in the generated image.
    # It is based on the gram matrices (which capture style) of
    # feature maps from the style reference image
    # and from the generated image
    def style_loss(style, combination):
        assert K.ndim(style) == 3
        assert K.ndim(combination) == 3

        S = gram_matrix(style)
        C = gram_matrix(combination)
        channels = 3
        size = img_nrows * img_ncols
        return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


    # an auxiliary loss function
    # designed to maintain the "content" of the
    # base image in the generated image
    def content_loss(base, combination):
        return K.sum(K.square(combination - base))



    # the 3rd loss function, total variation loss,
    # designed to keep the generated image locally coherent (no big changes)
    def total_variation_loss(x):
        assert K.ndim(x) == 4
        if K.image_data_format() == 'channels_first':

            a = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
            b = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
        else:
            a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
            b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
        return K.sum(K.pow(a + b, 1.25))

    loss = K.variable(0.)
    layer_features = outputs_dict[content_features[0]]
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss += content_weight * content_loss(base_image_features,
                                        combination_features)
    for layer_name in style_features:
        layer_features = outputs_dict[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(style_features)) * sl

    loss += total_variation_weight * total_variation_loss(combination_image)


    # ## Grandient Fucntions

    # In[66]:

    # get the gradients of the generated image wrt the loss

    grads = K.gradients(loss, combination_image)
    outputs = [loss]

    if isinstance(grads, (list, tuple)):
        outputs += grads
    else:
        outputs.append(grads)

    f_outputs = K.function([combination_image], outputs)

    def eval_loss_and_grads(x):

        if K.image_data_format() == 'channels_first':
            x = x.reshape((1, 3, img_nrows, img_ncols))
        else:
            x = x.reshape((1, img_nrows, img_ncols, 3))

        outs = f_outputs([x])
        loss_value = outs[0]
        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
        return loss_value, grad_values


    # ## Run Model

    # In[67]:

    # this Evaluator class makes it possible
    # to compute loss and gradients in one pass
    # while retrieving them via two separate functions,
    # "loss" and "grads". This is done because scipy.optimize
    # requires separate functions for loss and gradients,
    # but computing them separately would be inefficient.

    class Evaluator(object):

        def __init__(self):
            self.loss_value = None
            self.grads_values = None

        def loss(self, x):
            assert self.loss_value is None
            loss_value, grad_values = eval_loss_and_grads(x)
            self.loss_value = loss_value
            self.grad_values = grad_values
            return self.loss_value

        def grads(self, x):
            assert self.loss_value is not None
            grad_values = np.copy(self.grad_values)
            self.loss_value = None
            self.grad_values = None
            return grad_values
    evaluator = Evaluator()



    # run scipy-based optimization (L-BFGS) over the pixels of the generated image
    # so as to minimize the neural style loss
    x = preprocess_image(base_image_path)
    os.mkdir(trial_name)

    def write_config():
        fd = open(os.getcwd() + '/' + trial_name + '/' + 'weights.txt','w')
        fd.write('content_features\n')
        fd.write(config_cfeatures[w][0])
        fd.write('\nstyle features')
        for t in config_ffeatures[w]:
            fd.write(t)
            fd.write('\n')
        
        fd.close()
    write_config()
    # minimise the loss function
    for i in range(iterations):
        print('Start of iteration', i)
        start_time = time.time()

        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                        fprime=evaluator.grads, maxfun=20)
        
        # save current generated image
        img = deprocess_image(x.copy())
        fname = os.getcwd() +  '/' + trial_name + '/' + trial_name + '_at_iteration_%d.png' % i
        imsave(fname, img)
        end_time = time.time()

        print('Image saved as', fname)
        print('Iteration %d completed in %ds' % (i, end_time - start_time))



    # In[ ]:

    def generate_gif(pause):
        fns = [i for i in os.listdir(trial_name) if i[-4:] in ['.png','.jpeg']]
        imgs = []
        
        for fn in fns:
            i = 0
            while i < pause:
                imgs.append(imageio.imread(os.getcwd() + '/' + trial_name + '/' + fn))
                i += 1
        
        imageio.mimsave(os.getcwd() + '/' + trial_name + '/' + trial_name + '.gif',imgs)
            
    generate_gif(3) 
    print('h')
    w +=1
    print('k')

