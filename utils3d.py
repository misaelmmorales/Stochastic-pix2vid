import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from time import time                    
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

def check_tensorflow_gpu():
    sys_info = tf.sysconfig.get_build_info()
    print('Checking Tensorflow Version:')
    print('Tensorflow built with CUDA?',  tf.test.is_built_with_cuda())
    print('Tensorflow version:', tf.__version__)
    print('# GPU available:', len(tf.config.experimental.list_physical_devices('GPU')))
    print("CUDA: {} | cuDNN: {}".format(sys_info["cuda_version"], sys_info["cudnn_version"]))
    print(tf.config.list_physical_devices())
    return None