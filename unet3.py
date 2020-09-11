import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization, Conv2D, UpSampling2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import SGD, Adam, Adagrad
from tensorflow.keras import backend as K

import numpy as np
from numpy.random import seed
from datetime import datetime
from datetime import timedelta
import pickle
import os
import os.path
import math
import argparse


def accuracy05(y_true, y_pred):
    tp = tf.reduce_sum(tf.cast(tf.math.logical_and(y_true>0.5,y_pred>0.5), tf.float32))
    tn = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_true>0.5),tf.math.logical_not(y_pred>0.5)), tf.float32))

    return (tp+tn)/tf.cast(tf.size(y_true), tf.float32)
    

def precision05(y_true, y_pred):
    tp = tf.reduce_sum(tf.cast(tf.math.logical_and(y_true>0.5,y_pred>0.5), tf.float32))
    total_pred = tf.reduce_sum(tf.cast(y_pred>0.5, tf.float32))
    
    return tp/(total_pred+K.epsilon())


def recall05(y_true, y_pred):
    tp = tf.reduce_sum(tf.cast(tf.math.logical_and(y_true>0.5,y_pred>0.5), tf.float32))
    total_true = tf.reduce_sum(tf.cast(y_true>0.5, tf.float32))
    
    return tp/(total_true+K.epsilon())


def accuracy1(y_true, y_pred):
    tp = tf.reduce_sum(tf.cast(tf.math.logical_and(y_true>1,y_pred>1), tf.float32))
    tn = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_true>1),tf.math.logical_not(y_pred>1)), tf.float32))

    return (tp+tn)/tf.cast(tf.size(y_true), tf.float32)
    

def precision1(y_true, y_pred):
    tp = tf.reduce_sum(tf.cast(tf.math.logical_and(y_true>1,y_pred>1), tf.float32))
    #fp = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_true>1),y_pred>1), tf.float64))
    total_pred = tf.reduce_sum(tf.cast(y_pred>1, tf.float32))
    
    #if tf.math.less(total_pred, tf.constant([1.])):
    #    return 0.
    
    return tp/(total_pred+K.epsilon())


def recall1(y_true, y_pred):
    tp = tf.reduce_sum(tf.cast(tf.math.logical_and(y_true>1,y_pred>1), tf.float32))
    #fn = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_pred>1),y_true>1), tf.float64))
    total_true = tf.reduce_sum(tf.cast(y_true>1, tf.float32))
    
    #if tf.math.less(total_true, tf.constant([1.])):
    #    return 0.

    return tp/(total_true+K.epsilon())

def accuracy5(y_true, y_pred):
    tp = tf.reduce_sum(tf.cast(tf.math.logical_and(y_true>5,y_pred>5), tf.float32))
    tn = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_true>5),tf.math.logical_not(y_pred>5)), tf.float32))

    return (tp+tn)/tf.cast(tf.size(y_true), tf.float32)
    

def precision5(y_true, y_pred):
    tp = tf.reduce_sum(tf.cast(tf.math.logical_and(y_true>5,y_pred>5), tf.float32))
    #fp = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_true>5),y_pred>5), tf.float64))
    total_pred = tf.reduce_sum(tf.cast(y_pred>5, tf.float32))
    
    #if tf.math.less(total_pred, tf.constant([1.])):
    #    return 0.
    
    return tp/(total_pred+K.epsilon())


def recall5(y_true, y_pred):
    tp = tf.reduce_sum(tf.cast(tf.math.logical_and(y_true>5,y_pred>5), tf.float32))
    #fn = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_pred>5),y_true>5), tf.float64))
    total_true = tf.reduce_sum(tf.cast(y_true>5, tf.float32))
    
    #if tf.math.less(total_true, tf.constant([1.])):
    #    return 0.

    return tp/(total_true+K.epsilon())


def accuracy10(y_true, y_pred):
    tp = tf.reduce_sum(tf.cast(tf.math.logical_and(y_true>10,y_pred>10), tf.float32))
    tn = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_true>10),tf.math.logical_not(y_pred>10)), tf.float32))

    return (tp+tn)/tf.cast(tf.size(y_true), tf.float32)
    

def precision10(y_true, y_pred):
    tp = tf.reduce_sum(tf.cast(tf.math.logical_and(y_true>10,y_pred>10), tf.float32))
    #fp = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_true>10),y_pred>10), tf.float64))
    total_pred = tf.reduce_sum(tf.cast(y_pred>10, tf.float32))

    #if tf.math.less(total_pred, tf.constant([1.])):
    #    return 0.
    
    return tp/(total_pred+K.epsilon())


def recall10(y_true, y_pred):
    tp = tf.reduce_sum(tf.cast(tf.math.logical_and(y_true>10,y_pred>10), tf.float32))
    #fn = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_pred>10),y_true>10), tf.float64))
    total_true = tf.reduce_sum(tf.cast(y_true>10, tf.float32))
    
    #if tf.math.less(total_true, tf.constant([1])):
    #    return 0.

    return tp/(total_true+K.epsilon())


def get_unet():
    concat_axis = 3
    inputs = layers.Input(shape=(512, 512, 3))

    feats = 8#16
    bn0 = BatchNormalization(axis=3)(inputs)
    
    conv1 = layers.Conv2D(feats, (3, 3), activation='relu', padding='same', name='conv1_1')(bn0)
    bn2 = BatchNormalization(axis=3)(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(bn2) #256

    conv2 = layers.Conv2D(2*feats, (3, 3), activation='relu', padding='same')(pool1)
    bn4 = BatchNormalization(axis=3)(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(bn4) #128

    conv3 = layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same')(pool2)
    bn6 = BatchNormalization(axis=3)(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(bn6) #64

    conv4 = layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same')(pool3)
    bn8 = BatchNormalization(axis=3)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(bn8) #32

    conv5 = layers.Conv2D(16*feats, (3, 3), activation='relu', padding='same')(pool4)
    bn10 = BatchNormalization(axis=3)(conv5)
    pool5 = layers.MaxPooling2D(pool_size=(2, 2))(bn10) #16

    conv6 = layers.Conv2D(32*feats, (3, 3), activation='relu', padding='same')(pool5)
    bn11 = BatchNormalization(axis=3)(conv6)

    up_conv6 = layers.UpSampling2D(size=(2, 2))(bn11) #32
    up7 = layers.concatenate([up_conv6, conv5], axis=concat_axis)

    conv7 = layers.Conv2D(16*feats, (3, 3), activation='relu', padding='same')(up7)
    bn13 = BatchNormalization(axis=3)(conv7)
    
    up_conv5 = layers.UpSampling2D(size=(2, 2))(bn13) #64
    up6 = layers.concatenate([up_conv5, conv4], axis=concat_axis)

    conv6 = layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same')(up6)
    bn15 = BatchNormalization(axis=3)(conv6)

    up_conv6 = layers.UpSampling2D(size=(2, 2))(bn15) #128
    up7 = layers.concatenate([up_conv6, conv3], axis=concat_axis)
    
    conv7 = layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same')(up7)
    bn13 = BatchNormalization(axis=3)(conv7)

    # Rectify last convolution layer to constraint output to positive precipitation values.
    conv8 = layers.Conv2D(1, (1, 1), activation='relu')(bn13)

    model = models.Model(inputs=inputs, outputs=conv8)

    return model


def get_band_data(loc, dates, b, mean=None, std=None):
    y = np.concatenate([np.load(f"Y_{loc}_{d}.npy") for d in dates], axis=0)
    y = np.clip(y,0,30)
   
    x11 = np.concatenate([np.load(f"X_B11_{loc}_{d}.npy") for d in dates], axis=0)
    x16 = np.concatenate([np.load(f"X_B16_{loc}_{d}.npy") for d in dates], axis=0)
    xi = np.concatenate([np.load(f"X_B{b}_{loc}_{d}.npy") for d in dates], axis=0)
    
    if mean is None:
        mean = [x11.mean(),x16.mean(),xi.mean()]
        std = [x11.std(),x16.std(),xi.std()]
    
    x11 = (x11-mean[0])/std[0]
    x16 = (x16-mean[1])/std[1]
    xi = (xi-mean[2])/std[2]

    x = np.stack((x11,x16,xi), axis=3)
    x11 = None
    x16 = None
    xi = None
    
    return x, y[:,:,:,None], mean, std


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Himawari-GPM Band comparison')
    parser.add_argument('-b1', '--band1', help='Band 1 in list', type=int, required=True)
    parser.add_argument('-b2', '--band2', help='Band 2 in list', type=int, required=True)
    parser.add_argument('-b3', '--band3', help='Band 3 in list', type=int, required=True)
    parser.add_argument('-loc', '--location', help='Geographic location', type=str, required=True)
    parser.add_argument('-val', '--validation', help='Month used for validation', type=int, required=True)
    parser.add_argument('-s', '--seed', help='Random seed', type=int, required=False, default=1)
    args = parser.parse_args()

    seed(args.seed)
    
    if os.path.isfile(f'model_3months_200epochs_8chann_v{args.validation}_{args.location}_s{args.seed}_b{args.band1}_{args.band2}_{args.band3}.h5'):
        exit()
    
    tf.random.set_seed(args.seed)
   
    dates = ["201811","201812","201901","201902"]
    x_train, y_train, mean, std = get_band_data(args.location, [x for i, x in enumerate(dates) if i!=args.validation], args.band3)
    x_test, y_test, _, _ = get_band_data(args.location, [x for i, x in enumerate(dates) if i==args.validation], args.band3, mean, std)
 
    print(x_train.shape, y_train.shape)

    print("MSE train", np.mean(np.square(y_train)))
    print("MSE test", np.mean(np.square(y_test)))

    model = get_unet()
    print(model.summary())
    opt = Adagrad(lr=0.0001)
    model.compile(loss='mse', metrics=[accuracy05,precision05,recall05,accuracy1,precision1,recall1,accuracy5,precision5,recall5,accuracy10,precision10,recall10], optimizer=opt)
    
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), shuffle=True, epochs=200, verbose=1)
    
    with open(f'history_3months_200epochs_8chann_v{args.validation}_{args.location}_s{args.seed}_b{args.band1}_{args.band2}_{args.band3}.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    model.save(f'model_3months_200epochs_8chann_v{args.validation}_{args.location}_s{args.seed}_b{args.band1}_{args.band2}_{args.band3}.h5')
