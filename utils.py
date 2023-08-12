import os
from time import time
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from scipy.io import loadmat, savemat
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from skimage.metrics import mean_squared_error as img_mse
from skimage.metrics import structural_similarity as img_ssim

import tensorflow as tf
import keras.backend as K
from keras import Model, regularizers
from keras.layers import *
from tensorflow_addons.layers import *
from keras.optimizers import *
from keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.image import ssim as SSIM
from tensorflow.image import ssim_multiscale as MSSIM
from tensorflow.keras.callbacks import LearningRateScheduler

def check_tensorflow_gpu():
    sys_info = tf.sysconfig.get_build_info()
    print('Checking Tensorflow Version:')
    print('Tensorflow built with CUDA?',  tf.test.is_built_with_cuda())
    print('Tensorflow version:', tf.__version__)
    print('# GPU available:', len(tf.config.experimental.list_physical_devices('GPU')))
    print("CUDA: {} | cuDNN: {}".format(sys_info["cuda_version"], sys_info["cudnn_version"]))
    print(tf.config.list_physical_devices())
    return None

class SpatiotemporalCO2:
    def __init__(self):
        self.input_features_dir = 'simulations2D/input_features'
        self.output_targets_dir = 'simulations2D/output_targets'

        self.n_realizations = 1000
        self.x_channels = 4
        self.y_channels = 2
        self.timesteps  = 60
        self.dim        = 64
        self.test_size  = 0.25

        self.optimizer  = Nadam(learning_rate=1e-3)
        self.criterion  = self.custom_loss
        self.l2_alpha   = 0.9

        self.num_epochs = 100
        self.batch_size = 30
        self.lr_decay   = 10
        self.verbose    = 0

    def load_data(self):
        X_data = np.zeros((self.n_realizations, self.x_channels, self.dim, self.dim))
        y_data = np.zeros((self.n_realizations, self.timesteps, self.y_channels, self.dim, self.dim))
        for i in range(self.n_realizations):
            X_data[i] = np.load(self.input_features_dir + '/X_data_{}.npy'.format(i))
            y_data[i] = np.load(self.output_targets_dir + '/y_data_{}.npy'.format(i))
        self.X_data, self.y_data = np.moveaxis(X_data, 1, -1), np.moveaxis(y_data, 2, -1)
        print('X: {} | y: {}'.format(self.X_data.shape, self.y_data.shape))

    def process_data(self, subsample=None):
        num, height, width, channels = self.X_data.shape
        X_reshaped  = self.X_data.reshape(num*height*width, channels)
        X_scaled    = MinMaxScaler().fit_transform(X_reshaped)
        self.X_norm = X_scaled.reshape(num, height, width, channels)
        num, tsteps, height, width, channels = self.y_data.shape
        y_reshaped  = self.y_data.reshape(num*tsteps, -1)
        y_scaled    = MinMaxScaler().fit_transform(y_reshaped)
        self.y_norm = y_scaled.reshape(num, tsteps, height, width, channels)
        print('normalized - X: {} | y: {}'.format(self.X_norm.shape, self.y_norm.shape)) 
        if subsample != None:
            print('Subsampling data for {} samples ...'.format(subsample))
            idx = np.random.choice(range(self.n_realizations), subsample, replace=False)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_norm[idx], self.y_norm[idx], test_size=self.test_size)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_norm, self.y_norm, test_size=self.test_size)
        print('Train - X: {} | y: {}'.format(self.X_train.shape, self.y_train.shape))
        print('Test  - X: {} | y: {}'.format(self.X_test.shape, self.y_test.shape))

    def plot_loss(self, fit):
        ep = len(fit.history['loss'])
        it = np.arange(ep)
        plt.plot(it, fit.history['loss'], '-', label='loss')
        plt.plot(it, fit.history['val_loss'], '-', label='validation loss')
        plt.title('Training: Loss vs. Epochs'); plt.legend()
        plt.xlabel('Epochs'); plt.ylabel('Loss')
        plt.xticks(it[::ep//10]); plt.show()

    def custom_loss(self, true, pred):
        mse_loss = MeanSquaredError()(true, pred)
        ssim_loss = tf.reduce_mean(1.0 - SSIM(true, pred, max_val=1.0))
        combined_loss = self.l2_alpha * mse_loss + (1-self.l2_alpha) * ssim_loss
        return combined_loss

    class LossCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch+1) % 10 == 0:
                print('Epoch: {} - Loss: {:.4f} - Val Loss: {:.4f}'.format(epoch+1, logs['loss'], logs['val_loss']))

    def lr_scheduler(self, epoch, lr, schedule_type=1):
        if schedule_type == 1:
            if epoch % self.lr_decay == 0:
                new_lr = lr * 0.5
                return new_lr
        elif schedule_type == 2:
            if epoch < self.lr_decay:
                return lr
            else:
                return lr * tf.math.exp(-0.1)
        else:
            print('Select Type [1: halve every n epochs, 2: -0.1 exponential decay after n epochs]')
        return lr

    def encoder_layer(self, inp, filt, kern=(3,3), pool=(2,2), pad='same', leaky_slope=0.1):
        #x = SeparableConv2D(filters=filt, kernel_size=kern, padding=pad)(inp)
        #x = SeparableConv2D(filters=filt, kernel_size=kern, padding=pad)(x)
        #x = InstanceNormalization()(x)
        #x = GELU()(x)
        #x = AveragePooling2D(pool)(x)
        x = Conv2D(filters=filt, kernel_size=kern, padding=pad)(inp)
        x = BatchNormalization()(x)
        x = LeakyReLU(leaky_slope)(x)
        x = MaxPooling2D(pool)(x)
        return x

    def decoder_layer(self, inp, filt, kern=(3,3), pad='same', drop=0.1, leaky_slope=0.1):
        #x = ConvLSTM2D(filters=filt, kernel_size=kern, padding=pad, return_sequences=True, dropout=drop)(inp)
        #x = TimeDistributed(SeparableConv2D(filters=filt, kernel_size=kern, padding=pad))(x)
        #x = TimeDistributed(Conv2DTranspose(filters=filt, kernel_size=kern, padding=pad, strides=2))(x)
        #x = InstanceNormalization()(x)
        #x = GELU()(x)
        x = TimeDistributed(UpSampling2D())(inp)
        x = ConvLSTM2D(filters=filt, kernel_size=kern, padding=pad, return_sequences=True, dropout=drop)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(leaky_slope)(x)
        return x
       
    def make_model(self):
        inp = Input(shape=(self.dim, self.dim, self.x_channels))
        # Encoder
        _ = self.encoder_layer(inp, 16)
        _ = self.encoder_layer(_,   32)
        _ = self.encoder_layer(_,   64)
        # Recurrence
        _ = tf.expand_dims(_, 1)
        _ = tf.tile(_, (1, self.timesteps, 1, 1, 1))
        # Decoder
        _ = self.decoder_layer(_, 64)
        _ = self.decoder_layer(_, 32)
        _ = self.decoder_layer(_, 16)
        out = TimeDistributed(Conv2D(self.y_channels, (3,3), padding='same'))(_)
        # Output
        self.model = Model(inp, out)
        n_params = self.model.count_params()
        print('# Parameters: {:,}'.format(n_params))
        return self.model

    def train(self, model):
        model.compile(optimizer=self.optimizer, loss=self.criterion, metrics=['mse'])
        loss_cb = self.LossCallback()
        lr_schedule = LearningRateScheduler(self.lr_scheduler)
        if self.verbose==0:
            if self.lr_decay != None:
                cb = [loss_cb, lr_schedule]
            else:
                cb = [loss_cb]
        else:
            if self.lr_decay != None:
                cb = [lr_schedule]
            else:
                cb = []
        start = time()
        fit = model.fit(self.X_train, self.y_train,
                        epochs           = self.num_epochs,
                        batch_size       = self.batch_size,
                        validation_split = 0.20,
                        shuffle          = True,
                        callbacks        = cb,
                        verbose          = self.verbose)
        train_time = time()-start
        print('Training Time: {:.2f} minutes'.format(train_time/60))
        self.plot_loss(fit)
        return model, fit
