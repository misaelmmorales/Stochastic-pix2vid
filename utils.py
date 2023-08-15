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
from tensorflow.python.client import device_lib
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
    cuda_version, cudnn_version = sys_info['cuda_version'], sys_info['cudnn_version']
    num_gpu_avail = len(tf.config.experimental.list_physical_devices('GPU'))
    gpu_name = device_lib.list_local_devices()[1].physical_device_desc[17:40]
    print('... Checking Tensorflow Version ...')
    print('Tensorflow built with CUDA?',  tf.test.is_built_with_cuda())
    print("TF: {} | CUDA: {} | cuDNN: {}".format(tf.__version__, cuda_version, cudnn_version))
    print('# GPU available: {} ({})'.format(num_gpu_avail, gpu_name))
    #print(tf.config.list_physical_devices())
    return None

class SpatiotemporalCO2:
    def __init__(self):
        K.clear_session()
        self.input_features_dir = 'simulations2D/input_features'
        self.output_targets_dir = 'simulations2D/output_targets'

        self.n_realizations = 1000
        self.x_channels  = 4
        self.y_channels  = 2
        self.timesteps   = 60
        self.dim         = 64
        self.test_size   = 0.25
        self.t_samples   = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60]

        self.optimizer   = Nadam(learning_rate=1e-3)
        self.criterion   = self.custom_loss
        self.L1L2_split  = 0.25
        self.loss_alpha  = 0.8
        self.regular     = regularizers.l1(1e-4)
        self.leaky_slope = 0.25

        self.num_epochs  = 100
        self.batch_size  = 30
        self.lr_sch_type = 1
        self.lr_decay    = 25
        self.verbose     = 0

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
        print('MinMax Normalization Done!') 
        if subsample != None:
            print('Subsampling data for {} samples, {} timesteps ...'.format(subsample, len(self.t_samples)))
            idx = np.random.choice(range(self.n_realizations), subsample, replace=False)
            ts = np.array(self.t_samples); ts[1:]-=1
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_norm[idx], self.y_norm[idx][:,ts], test_size=self.test_size)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_norm, self.y_norm, test_size=self.test_size)
        print('Train - X: {} | y: {}'.format(self.X_train.shape, self.y_train.shape))
        print('Test  - X: {} | y: {}'.format(self.X_test.shape, self.y_test.shape))

    def plot_loss(self, fit, figsize=(4,3)):
        ep = len(fit.history['loss'])
        it = np.arange(ep)
        plt.figure(figsize=figsize)
        plt.plot(it, fit.history['loss'], '-', label='loss')
        plt.plot(it, fit.history['val_loss'], '-', label='validation loss')
        plt.title('Training: Loss vs. Epochs'); plt.legend()
        plt.xlabel('Epochs'); plt.ylabel('Loss')
        plt.xticks(it[::ep//10]); plt.show()

    def custom_loss(self, true, pred):
        ssim_loss = tf.reduce_mean(1.0 - SSIM(true, pred, max_val=1.0))
        if self.L1L2_split != None:
            mse_loss = MeanSquaredError()(true, pred)
            mae_loss = MeanAbsoluteError()(true,pred)
            ridge_loss = self.L1L2_split * mae_loss + (1-self.L1L2_split) * mse_loss
        else:
            ridge_loss = MeanSquaredError()(true, pred)
        combined_loss = self.loss_alpha * ridge_loss + (1-self.loss_alpha) * ssim_loss
        return combined_loss

    class LossCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch+1) % 10 == 0:
                print('Epoch: {} - Loss: {:.4f} - Val Loss: {:.4f}'.format(epoch+1, logs['loss'], logs['val_loss']))

    def lr_scheduler(self, epoch, lr):
        if self.lr_sch_type == 1:
            if epoch % self.lr_decay == 0:
                new_lr = lr * 0.5
                return new_lr
        elif self.lr_sch_type == 2:
            if epoch < self.lr_decay:
                return lr
            else:
                return lr * tf.math.exp(-0.1)
        else:
            print('Select Schedule Type [1: halve every n epochs, 2: -0.1 exponential decay after n epochs]')
        return lr

    def encoder_layer(self, inp, filt, kern=3, pad='same'):
        x = SeparableConv2D(filt, kern, padding=pad)(inp)
        xf = tf.math.real(tf.signal.fft2d(tf.cast(x, tf.complex64)))
        x = LayerNormalization()(x+xf)
        x = Conv2D(filt, kern, padding='same')(x)
        x = InstanceNormalization()(x)
        x = LeakyReLU(self.leaky_slope)(x)
        x = MaxPooling2D()(x)
        return x
    
    def recurrent_layer(self, inp, units, kern=3, pad='same'):
        x = tf.expand_dims(inp, 1)
        x = tf.tile(x, (1, self.y_train.shape[1], 1, 1, 1))
        x = ConvLSTM2D(units,   kern, padding=pad, return_sequences=True)(x)
        x = ConvLSTM2D(units*2, kern, padding=pad, return_sequences=True)(x)
        x = ConvLSTM2D(units,   kern, padding=pad, return_sequences=True)(x)
        return x

    def decoder_layer(self, inp, filt, kern=3, pad='same'):
        x = TimeDistributed(Conv2DTranspose(filt, kern, padding='same', strides=2))(inp)
        x = TimeDistributed(Conv2D(filt//2, kern, padding='same'))(x)
        xf = tf.math.real(tf.signal.fft2d(tf.cast(x, tf.complex64)))
        x = LayerNormalization()(x+xf)
        x = TimeDistributed(Conv2D(filt//2, kern, padding='same'))(x)
        x = InstanceNormalization()(x)
        x = LeakyReLU(self.leaky_slope)(x)
        return x
    
    def out_layer(self, inp, kern=3, pad='same', act='sigmoid'):
        x = TimeDistributed(Conv2D(self.y_channels, kern, padding=pad))(inp)
        #x = InstanceNormalization()(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(Activation(act))(x)
        return x
       
    def make_model(self):
        inputs = Input(shape=(self.dim, self.dim, self.x_channels))
        # Encoder
        _ = self.encoder_layer(inputs, 16)
        _ = self.encoder_layer(_,   64)
        z = self.encoder_layer(_,   128)
        # Recurrent
        z_inp = Input(z.shape[1:])
        zt = self.recurrent_layer(z_inp, 128)
        # Decoder
        _ = self.decoder_layer(zt, 128)
        _ = self.decoder_layer(_, 64)
        _ = self.decoder_layer(_, 16)
        # Output
        out = self.out_layer(_)
        # Models
        self.encoder = Model(inputs, z,  name='Encoder')
        self.decoder = Model(z_inp, out, name='Decoder')
        outputs = self.decoder(self.encoder(inputs))
        self.model   = Model(inputs, outputs, name='CNN-RNN-Proxy')
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
        n_params = self.model.count_params()
        print('# Parameters: {:,}'.format(n_params))
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


###################################################################################################
def plot_results(realization, X, ytrue, ypred, ts, cmap='jet', figsize=(20,4)):
    titles = ['Poro','LogPerm','Facies','Wells']
    plt.figure(figsize=(20,2))
    for i in range(4):
        plt.subplot(1,4,i+1)
        plt.imshow(X[realization,:,:,i], cmap=cmap)
        plt.xticks([]); plt.yticks([]); plt.title(titles[i], weight='bold')
    plt.show()

    fig, axs = plt.subplots(2, 11, figsize=(20,4))
    for j in range(11):
        axs[0,j].imshow(ytrue[realization,j,:,:,0], cmap=cmap)
        axs[1,j].imshow(ypred[realization,j,:,:,0], cmap=cmap)
        axs[0,j].set(title='t={}'.format(ts[j]))
        for i in range(2):
            axs[i,j].set(xticks=[], yticks=[])
    axs[0,0].set_ylabel('True', weight='bold'); axs[1,0].set_ylabel('Pred', weight='bold')
    plt.show()

    fig, axs = plt.subplots(2, 11, figsize=(20,4))
    for j in range(11):
        axs[0,j].imshow(ytrue[realization,j,:,:,1], cmap=cmap)
        axs[1,j].imshow(ypred[realization,j,:,:,1], cmap=cmap)
        axs[0,j].set(title='t={}'.format(ts[j]))
        for i in range(2):
            axs[i,j].set(xticks=[], yticks=[])
    axs[0,0].set_ylabel('True', weight='bold'); axs[1,0].set_ylabel('Pred', weight='bold')
    plt.show()
###################################################################################################
'''
    def encoder_layer(self, inp, filt, kern=2, pad='same'):
        x = SeparableConv2D(filt, kern, padding=pad, activation=LeakyReLU(self.leaky_slope))(inp)
        x = SeparableConv2D(filt, kern, padding=pad, kernel_regularizer=self.regular)(x)
        #x = InstanceNormalization()(x)
        x = LeakyReLU(self.leaky_slope)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D()(x)
        return x
    
    def decoder_layer(self, inp, filt, kern_s=(1,2,2), stride_s=(1,2,2), kern_t=(3,1,1), pad='same'):
        x = Conv3DTranspose(filt, kern_s, strides=stride_s, padding=pad, kernel_regularizer=self.regular)(inp)
        x = Conv3DTranspose(filt, kern_t, padding=pad, kernel_regularizer=self.regular)(x)
        #x = InstanceNormalization()(x)
        x = LeakyReLU(self.leaky_slope)(x)
        x = BatchNormalization()(x)
        return x 
    
    def recurrent_layer(self, inp, hidden_units, drop=0.1):
        height, width, channels = inp.shape[1:]
        time_steps = self.y_train.shape[1]
        x = tf.expand_dims(inp, -1)
        x = tf.tile(x, (1, 1, 1, 1, hidden_units))
        x = Reshape((np.prod(inp.shape[1:]), hidden_units))(x)
        x = LSTM(time_steps, return_sequences=True, dropout=drop)(x)
        x = LSTM(time_steps, return_sequences=True, dropout=drop)(x)
        x = tf.transpose(x, [0,2,1])
        x = Reshape((time_steps, height, width, channels))(x)
        return x
    
    def out_layer(self, inp, kern=(3,3), pad='same', act='sigmoid'):
        x = TimeDistributed(Conv2D(self.y_channels, kern, padding=pad))(inp)
        x = InstanceNormalization()(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(Activation(act))(x)
        return x
       
    def make_model(self):
        inputs = Input(shape=(self.dim, self.dim, self.x_channels))
        # Encoder
        _ = self.encoder_layer(inputs, 16)
        _ = self.encoder_layer(_,   64)
        z = self.encoder_layer(_,   128)
        # Recurrence
        z_inp = Input(z.shape[1:])
        zt = self.recurrent_layer(z_inp, 11)
        # Decoder
        zt_inp = Input(zt.shape[1:])
        _ = self.decoder_layer(zt_inp, 128)
        _ = self.decoder_layer(_, 64)
        _ = self.decoder_layer(_, 16)
        # Output
        out = self.out_layer(_)
        # Models
        self.encoder = Model(inputs, z,   name='Encoder')
        self.dynamic = Model(z_inp,  zt,  name='Dynamic')
        self.decoder = Model(zt_inp, out, name='Decoder')
        outputs = self.decoder(self.dynamic(self.encoder(inputs)))
        self.model   = Model(inputs, outputs, name='CNN-RNN-Proxy')
        return self.model

'''