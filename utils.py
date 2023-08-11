import os
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import loadmat, savemat
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from skimage.metrics import mean_squared_error as img_mse
from skimage.metrics import structural_similarity

import tensorflow as tf
import keras.backend as K
from keras import Model, regularizers
from keras.layers import *
from keras.optimizers import Adam
from keras.losses import mean_squared_error as loss_mse
from tensorflow_addons.layers import InstanceNormalization, GELU
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.image import ssim

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
        self.optimizer  = tf.keras.optimizers.Nadam(learning_rate=1e-3)
        self.loss_alpha = 0.5
        self.epochs     = 100
        self.batch_size = 30
        self.monitor_cb = 10

    def load_data(self):
        X_data = np.zeros((self.n_realizations, self.x_channels, self.dim, self.dim))
        y_data = np.zeros((self.n_realizations, self.timesteps, self.y_channels, self.dim, self.dim))
        for i in range(self.n_realizations):
            X_data[i] = np.load(self.input_features_dir + '/X_data_{}.npy'.format(i))
            y_data[i] = np.load(self.output_targets_dir + '/y_data_{}.npy'.format(i))
        self.X_data, self.y_data = np.moveaxis(X_data, 1, -1), np.moveaxis(y_data, 2, -1)
        print('X: {} | y: {}'.format(self.X_data.shape, self.y_data.shape))

    def process_data(self):
        num, height, width, channels = self.X_data.shape
        X_reshaped  = self.X_data.reshape(num*height*width, channels)
        X_scaled    = MinMaxScaler().fit_transform(X_reshaped)
        self.X_norm = X_scaled.reshape(num, height, width, channels)
        num, tsteps, height, width, channels = self.y_data.shape
        y_reshaped  = self.y_data.reshape(num*tsteps, -1)
        y_scaled    = MinMaxScaler().fit_transform(y_reshaped)
        self.y_norm = y_scaled.reshape(num, tsteps, height, width, channels)
        print('normalized - X: {} | y: {}'.format(self.X_norm.shape, self.y_norm.shape)) 
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_norm, self.y_norm, test_size=self.test_size)
        print('Train - X: {} | y: {}'.format(self.X_train.shape, self.y_train.shape))
        print('Test  - X: {} | y: {}'.format(self.X_test.shape, self.y_test.shape))

    def plot_loss(self):
        ep = len(self.fit.history['loss'])
        it = np.arange(ep)
        plt.plot(it, self.fit.history['loss'], '-', label='loss')
        plt.plot(it, self.fit.history['val_loss'], '-', label='validation loss')
        plt.title('Training: Loss vs. Epochs'); plt.legend()
        plt.xlabel('Epochs'); plt.ylabel('Loss')
        plt.xticks(it[::ep//10]); plt.show()

    def custom_loss(self, true, pred):
        mse = tf.keras.losses.MeanSquaredError()
        mse_loss = mse(true, pred)
        ssim_losses = [1.0 - ssim(true[:, t], pred[:, t], max_val=1.0) for t in range(true.shape[1])]
        ssim_loss = tf.reduce_mean(ssim_losses)
        combined_loss = self.loss_alpha * ssim_loss + (1-self.loss_alpha) * mse_loss
        return combined_loss

    class LossCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % self.monitor_cb == 0:
                print('Epoch: {} - Loss: {:.4f} - Val Loss: {:.4f}'.format(epoch, logs['loss'], logs['val_loss']))

    def lr_scheduler(self, epoch, lr):
        if epoch % self.monitor_cb == 0:
            new_lr = lr * 0.5
            return new_lr
        return lr

    def conv_block(self, inp, filt, kern=(3,3), pool=(2,2), pad='same'):
        x = SeparableConv2D(filters=filt, kernel_size=kern, padding=pad)(inp)
        x = SeparableConv2D(filters=filt, kernel_size=kern, padding=pad)(x)
        x = InstanceNormalization()(x)
        x = GELU()(x)
        x = AveragePooling2D(pool)(x)
        return x

    def decon_block(self, inp, filt, kern=(3,3), pad='same'):
        x = TimeDistributed(Conv2DTranspose(filters=filt, kernel_size=kern, padding=pad, strides=1))(inp)
        x = TimeDistributed(Conv2DTranspose(filters=filt, kernel_size=kern, padding=pad, strides=2))(x)
        x = InstanceNormalization()(x)
        x = GELU()(x)
        return x
    
    def make_model(self):
        K.clear_session()
        inp = Input(shape=(self.dim, self.dim, self.x_channels))
        # Encoder
        _ = self.conv_block(inp, 16)
        _ = self.conv_block(_,   32)
        z = self.conv_block(_,   64)
        # Recurrent
        _ = Flatten()(z)
        _ = RepeatVector(n=8*8*64)(_)
        _ = LSTM(units=60, return_sequences=True, dropout=0.2)(_)
        _ = tf.transpose(_, [0, 2, 1])
        zt = Reshape((60,8,8,64))(_)
        # Decoder
        _ = self.decon_block(zt, 64)
        _ = self.decon_block(_, 32)
        _ = self.decon_block(_, 16)
        out = TimeDistributed(Conv2D(filters=2, kernel_size=(3,3), padding='same'))(_)
        # Output
        self.encoder = Model(inp, z)
        self.dynamic = Model(z, zt)
        self.decoder_static  = Model(z, out)
        self.decoder_dynamic = Model(zt, out)
        self.model   = Model(inp, out)
        #
        n_params = self.model.count_params()
        print('# Parameters: {:,}'.format(n_params))
        return self.model

    def train(self):
        self.model.compile(optimizer=self.optimizer, loss=self.custom_loss, metrics=['mse'])
        loss_callback = self.LossCallback()
        lr_schedule = LearningRateScheduler(self.lr_scheduler)
        start = time()
        self.fit = self.model.fit(self.X_train, self.y_train,
                                    shuffle          = True,
                                    epochs           = self.epochs,
                                    validation_split = 0.20,
                                    batch_size       = self.batch_size,
                                    callbacks        = [loss_callback, lr_schedule],
                                    verbose          = 0)
        train_time = time()-start
        print('Training Time: {:.2f} minutes'.format(train_time/60))
        self.plot_loss()
        return self.fit
