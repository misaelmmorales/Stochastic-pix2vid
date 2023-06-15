import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import glob, os
from time import time 
import pyvista as pv
from scipy.io import loadmat, savemat
from skimage.transform import resize, rescale, downscale_local_mean
from sklearn.preprocessing import MinMaxScaler
 
from skimage.metrics import mean_squared_error as img_mse
from skimage.metrics import structural_similarity as img_ssim

import tensorflow as tf
import keras
import keras.backend as K
from keras import Model, regularizers
from keras.layers import Input
from keras.layers import BatchNormalization, LeakyReLU
from keras.layers import Flatten, Reshape, RepeatVector
from keras.layers import Conv2D, MaxPooling2D, Conv3D, Conv3DTranspose, GRU

dim      = 128    #size of images (128x128)
N_real   = 1000   #number of realizations
N_states = 75     #number of states

def check_tensorflow_gpu():
    sys_info = tf.sysconfig.get_build_info()
    print('Checking Tensorflow Version:')
    print('Tensorflow built with CUDA?',  tf.test.is_built_with_cuda())
    print('Tensorflow version:', tf.__version__)
    print('# GPU available:', len(tf.config.experimental.list_physical_devices('GPU')))
    print("CUDA: {} | cuDNN: {}".format(sys_info["cuda_version"], sys_info["cudnn_version"]))
    print(tf.config.list_physical_devices())
    return None

def load_process_data():
    # Load data from high-fidelity simulations
    poro_df       = loadmat('simulations/poro')['poro']              #porosity [frac]
    perm_df       = loadmat('simulations/perm')['perm']              #permeability[mD]
    saturation_df = loadmat('simulations/saturation')['saturation']  #saturation[frac]
    pressure_df   = loadmat('simulations/pressure')['pressure']      #pressure[Pascal]
    poro       = np.reshape(poro_df, [N_real,dim,dim])                  #porosity maps
    perm       = np.log10(np.reshape(perm_df, [N_real,dim,dim]))        #log10(perm) maps
    saturation = np.reshape(saturation_df, [N_real,dim,dim,N_states])   #saturation maps & states
    pressure   = np.reshape(pressure_df, [N_real,dim,dim,N_states])     #pressure maps & states
    print('Porosity shape: {}     | Permeability shape: {}'.format(poro.shape, perm.shape))
    print('Pressure shape: {} | Saturations  shape: {}'.format(pressure.shape, saturation.shape))
    return poro, perm, saturation, pressure

def augment_data(poro,perm,saturation,pressure):
    # rotate to make new images
    poro_flip        = np.rot90(poro, axes=(2,1))
    perm_flip        = np.rot90(perm, axes=(2,1))
    saturation_flip  = np.rot90(saturation, axes=(2,1))
    pressure_flip     = np.rot90(pressure, axes=(2,1))
    # concatenate original and rotated images & shuffle
    poro_cat         = np.concatenate((poro, poro_flip), axis=0)
    perm_cat         = np.concatenate((perm, perm_flip), axis=0)
    saturation_cat   = np.concatenate((saturation, saturation_flip), axis=0)
    pressure_cat     = np.concatenate((pressure, pressure_flip), axis=0)
    # Shuffle full dataset with (set) random index
    idx = np.random.choice(a=N_real*2, size=N_real*2, replace=False)
    poro_aug, perm_aug = np.zeros(poro_cat.shape), np.zeros(perm_cat.shape)
    saturation_aug = np.zeros(saturation_cat.shape)
    pressure_aug = np.zeros(pressure_cat.shape)
    for i in range(len(idx)):
        poro_aug[i,:,:] = poro_cat[idx[i],:,:]
        perm_aug[i,:,:] = perm_cat[idx[i],:,:]
        saturation_aug[i,:,:,:] = saturation_cat[idx[i],:,:,:]
        pressure_aug[i,:,:,:]   = pressure_cat[idx[i],:,:,:]
    print('Augmented Porosity shape:     {}'.format(poro_aug.shape)) 
    print('Augmented Permeability shape: {}'.format(perm_aug.shape)) 
    print('Augmented Saturation shape:   {}'.format(saturation_aug.shape))
    print('Augmented Pressure shape:     {}'.format(pressure_aug.shape))
    return poro_aug, perm_aug, saturation_aug, pressure_aug

# Function to plot a few samples from the (augmented) dataset
def plot_dataset(poro, perm, saturation, nrows=6, row_mult=75, dim=60, figsize=(20,10)):
    ncols, col_mult = 5, 12
    fig, axs = plt.subplots(nrows, ncols+3, figsize=figsize, tight_layout=True)
    for i in range(nrows):
        for j in range(ncols):
            # Plot saturation maps
            axs[i,j].imshow(saturation[i*row_mult,:,:,j*col_mult], cmap='plasma')
            axs[i,j].set(yticklabels=[], xticklabels=[], xticks=[], yticks=[])
            axs[i,0].set_ylabel('Realization {}'.format(i*row_mult))
            axs[0,j].set_title('State {}'.format(j*col_mult))
            # Plot (final state) saturation map
            axs[i,-3].imshow(saturation[i*row_mult,:,:,-1], cmap='plasma')
            axs[0,-3].set_title('State 60') 
            # Plot porosity map
            pom = axs[i,-2].imshow(poro[i*row_mult,:,:], cmap='viridis')
            axs[0,-2].set(title='Porosity')
            # Plot permeability map
            pem = axs[i,-1].imshow(perm[i*row_mult,:,:], cmap='jet')
            axs[0,-1].set(title='Log Permeability') 
        # Plot injector location
        for j in range(ncols+3):
            axs[i,j].plot(dim/2,dim/2, 'kh')
        for j in np.arange(ncols,ncols+3):
            axs[i,j].set(xticks=[], xticklabels=[], yticklabels=[])
        fig.colorbar(pom, ax=axs[i,-2], shrink=0.9)
        fig.colorbar(pem, ax=axs[i,-1], shrink=0.9)

# Function to plot training process
def plot_loss(fit):
    epochs     = len(fit.history['loss'])
    iterations = np.arange(epochs)
    plt.plot(iterations, fit.history['loss'],     '-', label='loss')
    plt.plot(iterations, fit.history['val_loss'], '-', label='validation loss')
    plt.title('Training: MSE vs epochs'); plt.legend()
    plt.xlabel('Epochs'); plt.ylabel('MSE')
    plt.xticks(iterations[::epochs//10])
    
# Function to plot to mean distributions of target and feature maps for train/test sets
def plot_histograms(X_training_flat, X_testing_flat, y_training_flat, y_testing_flat):
    plt.figure(figsize=(10,4))
    # Mean Permeability
    plt.subplot(121)
    plt.hist(X_training_flat.mean(axis=1), density=True, bins=15, label='train')
    plt.hist(X_testing_flat.mean(axis=1), density=True, bins=15, alpha=0.4, label='test')
    plt.title('Mean LogPerm Distribution - Train/Test')
    plt.xlabel('Mean Log Permeability'); plt.legend()
    # Mean End Saturation
    plt.subplot(122)
    plt.hist(y_training_flat.mean(axis=1), density=True, bins=15, label='train')
    plt.hist(y_testing_flat.mean(axis=1), density=True, bins=15, alpha=0.4, label='test')
    plt.title('Mean End Saturation Distribution - Train/Test')
    plt.xlabel('Mean End Saturation Permeability'); plt.legend()
    
# Function to visualize the true and predicted saturation maps for permeability realization
def compare_results_plot(X_test, y_true, y_pred, nrows=4, row_mult=40, figsize=(20,7)):
    month_states = [0, 11, 23, 35, 47, 59]
    fig, axs = plt.subplots(4, 13, figsize=figsize, tight_layout=True)
    for i in range(nrows):
        for j in np.arange(len(month_states)):
            # Show true saturation states (y_test)
            axs[i,j].imshow(y_true[i*row_mult, :, :, month_states[j]], cmap='plasma')
            axs[i,j].set(yticklabels=[], xticklabels=[], xticks=[], yticks=[])
            # Show predicted saturation sates (y_pred)
            k = j+7
            axs[i,k].imshow(y_pred[i*row_mult, :, :, month_states[j]], cmap='plasma')
            axs[i,k].set(yticklabels=[], xticklabels=[], xticks=[], yticks=[])
            # Add axis labels
            axs[i,0].set_ylabel('Realization {}'.format(i*row_mult))
            axs[0,j].set_title('State {}'.format(month_states[j]+1))
            axs[0,k].set_title('State {}'.format(month_states[j]+1))
        # Show permeability map (X)
        axs[i,len(month_states)].imshow(X_test[i*row_mult,:,:], cmap='jet')
        axs[i,len(month_states)].set(yticklabels=[], xticklabels=[], yticks=[])
        axs[0,len(month_states)].set_title('Log Perm')
