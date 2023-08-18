import tensorflow as tf
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

import numpy as np
import random
import os
os.environ['PYTHONHASHSEED']=str(42)


from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda, Activation, GlobalAveragePooling2D
from keras.layers import Activation, MaxPool2D, Concatenate, add , multiply, Reshape, Dense, Permute, GlobalMaxPooling2D, Add, Conv1D
from keras import backend as K
from tensorflow.keras import models, layers, regularizers

def dice_coef(y_true, y_pred, smooth=100):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return dice

def dice_coef_loss(y_true, y_pred):
        return -dice_coef(y_true, y_pred)


def iou(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        def f(y_true, y_pred):
            intersection = (y_true * y_pred).sum()
            union = y_true.sum() + y_pred.sum() - intersection
            x = (intersection + 1e-15) / (union + 1e-15)
            x = x.astype(np.float32)
            return x
        
        return tf.numpy_function(f, [y_true, y_pred], tf.float32)

def MultiResBlock(input, num_filters):
    W = num_filters
    
    shortcut = input
    shortcut = Conv2D(int(W*0.333) + int(W*0.333) + int(W*0.333), 1, padding='same')(shortcut)
    
    x1_1 = Conv2D(int(W*0.333), 3, padding="same", dilation_rate=1)(input)
    x1_1 = BatchNormalization(axis=3)(x1_1)
    x1_1 = Activation("relu")(x1_1)
    
    x1_2 = Conv2D(int(W*0.333), 3, padding="same", dilation_rate=3)(input)
    x1_2 = BatchNormalization(axis=3)(x1_2)
    x1_2 = Activation("relu")(x1_2)
    
    x1_3 = Conv2D(int(W*0.333), 3, padding="same", dilation_rate=5)(input)
    x1_3 = BatchNormalization(axis=3)(x1_3) 
    x1_3 = Activation("relu")(x1_3)
    
    x = concatenate([x1_1,x1_2,x1_3], axis=3)
    x = BatchNormalization(axis=3)(x)
    
    out = add([shortcut, x])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)
    return out

def ResPath(filters, length, inp):
    shortcut = inp
    shortcut = Conv2D(filters, 1, padding='same')(shortcut)
    shortcut = BatchNormalization(axis=3)(shortcut)
    
    out = Conv2D(filters, 3, padding='same')(inp)
    out = BatchNormalization(axis=3)(out)
    out = Activation("relu")(out)
    
    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    for i in range(length-1):

        shortcut = out
        shortcut = Conv2D(filters, 1, padding='same')(shortcut)
        shortcut = BatchNormalization(axis=3)(shortcut)

        out = Conv2D(filters, 3, padding='same')(out)
        out = BatchNormalization(axis=3)(out)
        out = Activation("relu")(out)

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization(axis=3)(out)

    return out

def channel_attention(input_feature, ratio=4):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]

    shared_layer_one = Dense(channel//ratio,
                            activation='relu',
                            kernel_initializer='he_normal',
                            use_bias=True,
                            bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                            kernel_initializer='he_normal',
                            use_bias=True,
                            bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)    
    avg_pool = Reshape((1,1,channel))(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1,1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)

    cbam_feature = Add()([avg_pool,max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])

def custom_unetv20(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = Lambda(lambda x: x / 255)(inputs) 
    s = inputs

    # Downsampling layers
    c1 = MultiResBlock(s, 16)
    p1 = MaxPool2D((2,2))(c1)
    c1 = ResPath(16, 4, c1)
    
    c2 = MultiResBlock(p1, 32)
    p2 = MaxPool2D((2,2))(c2)
    c2 = ResPath(16*2, 3, c2)
    
    c3 = MultiResBlock(p2, 64)
    p3 = MaxPool2D((2,2))(c3)
    c3 = ResPath(16*4, 2, c3)
    
    c4 = MultiResBlock(p3, 128)
    p4 = MaxPool2D((2,2))(c4)
    c4 = ResPath(16*8, 1, c4)
    
    c5 = MultiResBlock(p4, 256)
    c5 = channel_attention(c5)

    # Upsampling layers
    u1 = Conv2DTranspose(128, (2, 2), strides=2, padding="same")(c5)
    u1 = concatenate([u1, c4], axis=3)
    up_c4 = MultiResBlock(u1, 128)
 
    u2 = Conv2DTranspose(64, (2, 2), strides=2, padding="same")(up_c4)
    u2 = concatenate([u2, c3], axis=3)
    up_c3 = MultiResBlock(u2, 64)

    u3 = Conv2DTranspose(32, (2, 2), strides=2, padding="same")(up_c3)
    u3 = concatenate([u3, c2], axis=3)
    up_c2 = MultiResBlock(u3, 32)

    u4 = Conv2DTranspose(16, (2, 2), strides=2, padding="same")(up_c2)
    u4 = concatenate([u4, c1], axis=3)
    up_c1 = MultiResBlock(u4, 16)

    output = Conv2D(1, kernel_size=(1,1))(up_c1)
    output = BatchNormalization(axis=3)(output)
    output = Activation('sigmoid')(output)

    model = Model(inputs=[inputs], outputs=[output])
    model.compile(optimizer='adam', loss = [dice_coef_loss], metrics = [iou])
    model.summary()
    
    return model

