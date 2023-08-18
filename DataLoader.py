#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()


# In[2]:


import numpy as np
import random
import os
os.environ['PYTHONHASHSEED']=str(42)


# In[3]:


from tqdm import tqdm 

from skimage.io import imread, imshow
from skimage.transform import resize


# In[4]:


IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3


# In[5]:


IMAGE_PATH = 'new_data/images/'
MASK_PATH = 'new_data/masks/'


# In[6]:


image_ids = next(os.walk(IMAGE_PATH))[2]
mask_ids = next(os.walk(MASK_PATH))[2]

image_data = np.zeros((len(image_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
mask_data = np.zeros((len(mask_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool_)


# In[7]:


np.shape(image_data)


# In[8]:


def DataLoader():
    print('Loading Images')
    for n, image in tqdm(enumerate(image_ids), total=len(image_ids)):   
        img = imread('new_data/images/' + image)[:,:,:IMG_CHANNELS]  
        image_data[n] = img 
    
    print('Loading Masks')
    for n, mask in tqdm(enumerate(mask_ids), total=len(mask_ids)):   
        mask = imread('new_data/masks/' + mask)[:,:,:1]
        mask_data[n] = mask
    return image_data, mask_data

