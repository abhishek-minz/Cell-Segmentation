#!/usr/bin/env python
# coding: utf-8

# ## Data Merging

# In[1]:


import tensorflow as tf
import os
import random
import numpy as np
 
from tqdm import tqdm 

from skimage.io import imread, imshow, imsave
from skimage.transform import resize
import matplotlib.pyplot as plt


# In[2]:


seed_value= 42
os.environ['PYTHONHASHSEED']=str(seed_value)
#random.seed(seed_value)

np.random.seed = seed_value
tf.random.set_seed(seed_value)
# tf.set_random_seed(seed_value)


# In[3]:


IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3


# In[4]:


TRAIN_PATH = 'stage1_train/'


# In[5]:


train_ids = next(os.walk(TRAIN_PATH))[1]

image_data = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
mask_data = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool_)


# In[7]:


print('Resizing training images and masks')
os.mkdir('data')
os.mkdir('data/images/')
os.mkdir('data/masks/')
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]  
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    image_data[n] = img  #Fill empty image_data with values from img
    imsave('data/images/' + id_ + ".png", (img).astype(np.uint8))
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool_)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    imsave('data/masks/' + id_ + ".png", np.squeeze(mask).astype(bool))
    mask_data[n] = mask 


# In[ ]:




