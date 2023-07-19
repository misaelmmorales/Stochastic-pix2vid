import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys, os, glob,cv2
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

import torch
import torch.nn as nn
from torch.nn import Sequential
from torch.nn import Linear, ReLU, LeakyReLU, Dropout, BatchNorm1d
from torch.nn import Conv2d, ConvTranspose2d, MaxPool2d, BatchNorm2d
from torch.optim import NAdam, Adam
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn.functional as F

dim      = 128    #size of images (128x128)
N_real   = 1000   #number of realizations
N_states = 50     #number of states

def check_tensorflow_gpu():
    sys_info = tf.sysconfig.get_build_info()
    print('Checking Tensorflow Version:')
    print('Tensorflow built with CUDA?',  tf.test.is_built_with_cuda())
    print('Tensorflow version:', tf.__version__)
    print('# GPU available:', len(tf.config.experimental.list_physical_devices('GPU')))
    print("CUDA: {} | cuDNN: {}".format(sys_info["cuda_version"], sys_info["cudnn_version"]))
    print(tf.config.list_physical_devices())
    return None

def check_torch_gpu():
    '''
    Check torch build in python to ensure GPU is available for training.
    '''
    torch_version, cuda_avail = torch.__version__, torch.cuda.is_available()
    count, name = torch.cuda.device_count(), torch.cuda.get_device_name()
    py_version, conda_env_name = sys.version, sys.executable.split('\\')[-2]
    print('-------------------------------------------------')
    print('------------------ VERSION INFO -----------------')
    print('Conda Environment: {} | Python version: {}'.format(conda_env_name, py_version))
    print('Torch version: {}'.format(torch_version))
    print('Torch build with CUDA? {}'.format(cuda_avail))
    print('# Device(s) available: {}, Name(s): {}\n'.format(count, name))
    device = torch.device('cuda' if cuda_avail else 'cpu')
    return None