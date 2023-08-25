import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation

from scipy.io import loadmat, savemat
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from skimage.metrics import mean_squared_error as img_mse
from skimage.metrics import structural_similarity as img_ssim

import keras.backend as K
from keras import Model, regularizers
from keras.layers import *
from keras.optimizers import SGD, Adam, Nadam
from keras.losses import MeanSquaredError, MeanAbsoluteError

import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow_addons.layers import *
from tensorflow_addons.optimizers import AdamW
from tensorflow.image import ssim as SSIM
from tensorflow.keras.metrics import mean_squared_error as MSE
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau

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

class SpectralProduct(Layer):
    def __init__(self, **kwargs):
        super(SpectralProduct, self).__init__(**kwargs)
    def call(self, inputs):
        x0 = tf.transpose(inputs, [0,3,1,2])
        xf = tf.signal.rfft2d(x0)
        xr = tf.einsum('ijkl, ijpq -> ijpq', x0, tf.math.real(xf))
        xi = tf.einsum('ijkl, ijpq -> ijpq', x0, tf.math.imag(xf))
        xc = tf.complex(xr, xi)
        x  = tf.signal.irfft2d(xc)
        x  = tf.transpose(x, [0,2,3,1])
        return x
    
class SqueezeExcite(Layer):
    def __init__(self, ratio=4, **kwargs):
        super(SqueezeExcite, self).__init__(**kwargs)
        self.ratio = ratio
    def build(self, input_shape):
        channels = input_shape[-1]
        self.squeeze = GlobalAveragePooling2D()
        self.excite1 = Dense(channels // self.ratio, activation='relu')
        self.excite2 = Dense(channels, activation='sigmoid')
        super(SqueezeExcite, self).build(input_shape)
    def call(self, inputs):
        se_tensor = self.squeeze(inputs)
        se_tensor = self.excite1(se_tensor)
        se_tensor = self.excite2(se_tensor)
        se_tensor = Reshape((1, 1, se_tensor.shape[-1]))(se_tensor)
        scaled_inputs = Multiply()([inputs, se_tensor])
        return Add()([inputs, scaled_inputs])
    def compute_output_shape(self, input_shape):
        return input_shape

class SpatiotemporalCO2:
    def __init__(self):
        K.clear_session()
        self.input_features_dir = 'simulations2D/input_features'
        self.output_targets_dir = 'simulations2D/output_targets'
        self.x_data_labels = ['Poro','LogPerm','Facies','Wells']
        self.y_data_labels = ['Pressure', 'Saturation']
        self.return_data = False
        self.save_model  = False

        self.n_realizations = 1000
        self.x_channels  = 4
        self.y_channels  = 2
        self.timesteps   = 60
        self.dim         = 64
        self.test_size   = 0.25
        self.t_samples   = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
        
        self.cnn_filters = [16,  64, 256]
        self.rnn_filters = [256, 64, 16]
        self.rnn_dropout = 0.2
        self.up_interp   = 'bilinear'

        #SGD(learning_rate=1e-3) #Nadam(learning_rate=1e-3)
        self.optimizer   = AdamW(learning_rate=1e-3, weight_decay=1e-6)
        self.criterion   = self.custom_loss
        self.L1L2_split  = 0.25
        self.ridge_alpha = 0.70
        self.regular     = regularizers.l1(1e-6)
        self.leaky_slope = 0.25

        self.num_epochs  = 200
        self.batch_size  = 60
        self.lr_decay    = 15
        self.verbose     = 0

    def load_data(self):
        X_data = np.zeros((self.n_realizations, self.x_channels, self.dim, self.dim))
        y_data = np.zeros((self.n_realizations, self.timesteps, self.y_channels, self.dim, self.dim))
        for i in range(self.n_realizations):
            X_data[i] = np.load(self.input_features_dir + '/X_data_{}.npy'.format(i))
            y_data[i] = np.load(self.output_targets_dir + '/y_data_{}.npy'.format(i))
        self.X_data, self.y_data = np.moveaxis(X_data, 1, -1), np.moveaxis(y_data, 2, -1)
        print('X: {} | y: {}'.format(self.X_data.shape, self.y_data.shape))

    def process_data(self, n_subsample=None, augment=True, rots=3):
        # data augmentation
        if augment==True:
            xrot = np.rot90(self.X_data, k=rots, axes=(1,2))
            yrot = np.rot90(self.y_data, k=rots, axes=(2,3))
            x = np.concatenate([self.X_data, xrot], axis=0)
            y = np.concatenate([self.y_data, yrot], axis=0)
            print('Data Augmentation Done! - n_samples={:,}'.format(x.shape[0]))
        else:
            x = self.X_data
            y = self.y_data
        # feature processing
        num, height, width, channels = x.shape
        X_reshaped  = x.reshape(num*height*width, channels)
        self.X_norm = MinMaxScaler().fit_transform(X_reshaped).reshape(num, height, width, channels)
        # target processing
        num, tsteps, height, width, channels = y.shape
        y_reshaped  = y.reshape(num*tsteps, -1)
        self.y_norm = MinMaxScaler().fit_transform(y_reshaped).reshape(num, tsteps, height, width, channels)
        print('MinMax Normalization Done! - [{}, {}]'.format(self.X_norm.min(), self.X_norm.max()))
        # train-test split and subsampling
        ts = np.array(self.t_samples); ts[1:]-=1
        if n_subsample != None:
            print('Subsampling data for {} samples, {} timesteps ...'.format(n_subsample, len(self.t_samples)))
            idx = np.random.choice(range(self.X_norm.shape[0]), n_subsample, replace=False)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_norm[idx], self.y_norm[idx][:,ts], test_size=self.test_size)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_norm, self.y_norm[:,ts], test_size=self.test_size)
        print('Train - X: {} | y: {}'.format(self.X_train.shape, self.y_train.shape))
        print('Test  - X: {} | y: {}'.format(self.X_test.shape, self.y_test.shape))

    def plot_features(self, nsamples=10, multiplier=1, figsize=(15,5), cmaps=['jet','jet','viridis','binary']):
        _, axs = plt.subplots(4, nsamples, figsize=figsize)
        for i in range(4):
            for j in range(nsamples):
                axs[i,j].imshow(self.X_norm[j*multiplier,:,:,i], cmap=cmaps[i])
                axs[i,j].set(xticks=[], yticks=[])
                axs[i,0].set_ylabel(self.x_data_labels[i], weight='bold')
                axs[0,j].set_title('R{}'.format(j*multiplier), weight='bold')
        plt.show()
    
    def plot_targets(self, p_s=1, nsamples=10, multiplier=1, figsize=(15,10), cmap='jet'):
        ts = np.array(self.t_samples); ts[1:]-=1
        _, axs = plt.subplots(len(ts), nsamples, figsize=figsize)
        for i in range(len(ts)):
            for j in range(nsamples):
                axs[i,j].imshow(self.y_norm[j*multiplier, ts[i], :, :, p_s], cmap=cmap)
                axs[i,j].set(xticks=[], yticks=[])
                axs[0,j].set_title('R{}'.format(j*multiplier), weight='bold')
                axs[i,0].set_ylabel('t={}'.format(ts[i]+1), weight='bold')
        plt.show()
    
    def plot_data(self, nsamples=10, multiplier=1, p_s=1, cmaps=['jet','jet','viridis','binary','jet'], figsize=(12,15)):
        ts = np.array(self.t_samples); ts[1:]-=1
        nx, ny = self.x_channels, len(ts)
        _, axs = plt.subplots(nx+ny, nsamples, figsize=figsize, tight_layout=True)
        # features
        for i in range(nx):
            for j in range(nsamples):
                axs[i,j].imshow(self.X_norm[j*multiplier, :, :, i], cmap=cmaps[i])
            axs[i,0].set_ylabel(self.x_data_labels[i], weight='bold')
        # targets
        for i in range(ny):
            for j in range(nsamples):
                p = i+nx
                axs[p,j].imshow(self.y_norm[j*multiplier, ts[i], :, :, p_s], cmap=cmaps[-1])
            axs[p,0].set_ylabel('t={}'.format(ts[i]+1), weight='bold')
        # plotting
        for j in range(nsamples):
            axs[0,j].set_title('R{}'.format(j*multiplier), weight='bold')
            for i in range(nx+ny):
                axs[i,j].set(xticks=[], yticks=[])
        plt.show()

    def plot_loss(self, figsize=(5,3), cs=['tab:blue','tab:orange'], grid=True):
        ep = len(self.fit.history['loss'])
        it = np.arange(ep)
        plt.figure(figsize=figsize)
        plt.plot(it, self.fit.history['loss'], c=cs[0], linestyle='-', label='loss')
        plt.plot(it, self.fit.history['val_loss'], c=cs[1], linestyle='-', label='validation loss')
        plt.title('Training: Loss vs. Epochs'); plt.xlabel('Epochs'); plt.ylabel('Loss')
        plt.legend(); plt.xticks(it[::ep//10])
        if grid==True:
            plt.grid('on')
        plt.show()

    def custom_loss(self, true, pred):
        ssim_loss = tf.reduce_mean(1.0 - SSIM(true, pred, max_val=1.0))
        if self.L1L2_split != None:
            mse_loss = MeanSquaredError()(true, pred)
            mae_loss = MeanAbsoluteError()(true,pred)
            ridge_loss = self.L1L2_split * mae_loss + (1-self.L1L2_split) * mse_loss
        else:
            ridge_loss = MeanSquaredError()(true, pred)
        combined_loss = self.ridge_alpha * ridge_loss + (1-self.ridge_alpha) * ssim_loss
        return combined_loss

    class LossCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch+1) % 10 == 0:
                print('Epoch: [{}/{}] - Loss: {:.4f} - Val Loss: {:.4f}'.format(epoch+1, SpatiotemporalCO2().num_epochs, logs['loss'], logs['val_loss']))

    def encoder_layer(self, inp, filt, kern=3, pad='same'):
        _ = SeparableConv2D(filt, kern, padding=pad, activity_regularizer=self.regular, depth_multiplier=2)(inp)
        _ = SqueezeExcite()(_)
        _ = InstanceNormalization()(_)
        _ = PReLU()(_)
        _ = MaxPooling2D()(_)
        return _
    
    def recurrent_decoder(self, z_input, residuals, previous_timestep=None):
        dropout = self.rnn_dropout
        def recurrent_step(inp, filt, res, kern=3, pad='same', drop=dropout):
            y = ConvLSTM2D(filt, kern, padding=pad, dropout=drop)(inp)
            y = BatchNormalization()(y)
            y = LeakyReLU(self.leaky_slope)(y)
            #y = UpSampling2D(interpolation=self.up_interp)(y)
            y = Conv2DTranspose(filt, kern, strides=2, padding='same', activity_regularizer=self.regular)(y)
            y = Concatenate()([y, res])
            y = Conv2D(filt, kern, padding=pad)(y)
            y = Activation('sigmoid')(y)
            y = tf.expand_dims(y,1)
            return y
        def recurrent_last(inp, filt, kern=3, pad='same', drop=dropout):
            y = ConvLSTM2D(filt, kern, padding=pad, dropout=drop)(inp)
            y = BatchNormalization()(y)
            y = LeakyReLU(self.leaky_slope)(y)
            #y = UpSampling2D(interpolation=self.up_interp)(y)
            #y = Conv2D(self.y_channels, kern, padding=pad, groups=2)(y)
            y = Conv2DTranspose(self.y_channels, kern, strides=2, padding='same', activity_regularizer=self.regular)(y)
            y = Activation('sigmoid')(y)
            y = tf.expand_dims(y, 1)
            return y
        # recurrent-decoder architecture
        #res3, res2 = residuals
        #filt3, filt2, filt1 = self.rnn_filters
        _ = tf.expand_dims(z_input, 1)
        _ = recurrent_step(_, self.rnn_filters[0], residuals[0])
        _ = recurrent_step(_, self.rnn_filters[1], residuals[1])
        _ = recurrent_last(_, self.rnn_filters[2])
        if previous_timestep != None:
            _ = Concatenate(axis=1)([previous_timestep, _])
        return _
    
    def make_model(self):
        inp = Input(self.X_train.shape[1:])
        z1  = self.encoder_layer(inp, self.cnn_filters[0])
        z2  = self.encoder_layer(z1,  self.cnn_filters[1])
        z3  = self.encoder_layer(z2,  self.cnn_filters[2])
        t0  = self.recurrent_decoder(z3, [z2,z1])
        t1  = self.recurrent_decoder(z3, [z2,z1], previous_timestep=t0)
        t2  = self.recurrent_decoder(z3, [z2,z1], previous_timestep=t1)
        t3  = self.recurrent_decoder(z3, [z2,z1], previous_timestep=t2)
        t4  = self.recurrent_decoder(z3, [z2,z1], previous_timestep=t3)
        t5  = self.recurrent_decoder(z3, [z2,z1], previous_timestep=t4)
        t6  = self.recurrent_decoder(z3, [z2,z1], previous_timestep=t5)
        t7  = self.recurrent_decoder(z3, [z2,z1], previous_timestep=t6)
        t8  = self.recurrent_decoder(z3, [z2,z1], previous_timestep=t7)
        t9  = self.recurrent_decoder(z3, [z2,z1], previous_timestep=t8)
        t10 = self.recurrent_decoder(z3, [z2,z1], previous_timestep=t9)
        self.latent1 = Model(inp, z1, name='enc_layer_1'); self.latent1.compile('adam','mse',['mse'])
        self.latent2 = Model(inp, z2, name='enc_layer_2'); self.latent2.compile('adam','mse',['mse'])
        self.encoder = Model(inp, z3, name='encoder');     self.encoder.compile('adam','mse',['mse'])
        self.model   = Model(inp, t10, name='CNN_RNN_Proxy')
        if self.return_data:
            return self.model, self.encoder

    def training(self):
        self.model.compile(optimizer=self.optimizer, loss=self.criterion, metrics=['mse'])
        loss_cb     = self.LossCallback()
        lr_reduce   = ReduceLROnPlateau(patience=self.lr_decay) 
        if self.verbose==0:
            if self.lr_decay != None:
                cb = [loss_cb, lr_reduce]
            else:
                cb = [loss_cb]
        else:
            if self.lr_decay != None:
                cb = [lr_reduce]
            else:
                cb = []
        n_params = self.model.count_params()
        print('# Parameters: {:,} | Batch size: {} '.format(n_params, self.batch_size))
        start = time()
        self.fit = self.model.fit(self.X_train, self.y_train,
                                    epochs           = self.num_epochs,
                                    batch_size       = self.batch_size,
                                    validation_split = 0.20,
                                    shuffle          = True,
                                    callbacks        = cb,
                                    verbose          = self.verbose)
        train_time = time()-start
        print('Training Time: {:.2f} minutes'.format(train_time/60))
        self.plot_loss()
        if self.save_model:
            self.model.save('cnn_rnn_proxy')
        if self.return_data:
            return self.model, self.fit

    def compute_mean_ssim(self, true, pred):
        mean_ssim = {}
        for i in range(true.shape[1]):
            mean_ssim[i] = SSIM(true[:,i], pred[:,i], max_val=1.0)
        return np.array(list(mean_ssim.values())).mean(0).mean()
    
    def compute_mean_mse(self, true, pred):
        mean_mse = {}
        for i in range(true.shape[1]):
            mean_mse[i] = MSE(true[:,i], pred[:,i])
        return np.array(list(mean_mse.values())).mean(0).mean()
    
    def predictions(self):
        self.y_train_pred = self.model.predict(self.X_train).astype('float64')
        self.y_test_pred  = self.model.predict(self.X_test).astype('float64')
        print('Train pred: {} | Test pred: {}'.format(self.y_train_pred.shape, self.y_test_pred.shape))
        train_mse  = self.compute_mean_mse(self.y_train,  self.y_train_pred)
        train_ssim = self.compute_mean_ssim(self.y_train, self.y_train_pred)
        test_mse   = self.compute_mean_mse(self.y_test,   self.y_test_pred)
        test_ssim  = self.compute_mean_ssim(self.y_test,  self.y_test_pred)
        print('MSE  | Train: {:.2e}, Test: {:.2e}'.format(train_mse, test_mse))
        print('SSIM | Train: {:.2f}, Test: {:.2f}'.format(train_ssim*100, test_ssim*100))
        if self.return_data:
            return self.y_train_pred, self.y_test_pred
    
    def plot_single_results(self, realization, train_or_test='train', cmaps=['jet','jet','viridis','binary','binary'], figsize=(20,4)):
        labels = ['True', 'Predicted', 'Abs. Error']
        if train_or_test == 'train':
            x, y, yhat = self.X_train, self.y_train, self.y_train_pred
        elif train_or_test == 'test':
            x, y, yhat = self.X_test, self.y_test, self.y_test_pred
        else:
            print('Please select "train" or "test" to display')
        # Geologic model (static inputs)
        plt.figure(figsize=figsize)
        for i in range(4):
            plt.subplot(1,4,i+1)
            plt.imshow(x[realization,:,:,i], cmap=cmaps[i])
            plt.xticks([]); plt.yticks([]); plt.title(self.x_data_labels[i], weight='bold')
        plt.suptitle('Geologic Models (static inputs) - R{}'.format(realization), weight='bold'); plt.show()
        # Pressure results (dynamic output 0)
        fig, axs = plt.subplots(3, len(self.t_samples), figsize=figsize)
        for j in range(len(self.t_samples)):
            true, pred = y[realization,j,:,:,0], yhat[realization,j,:,:,0]
            axs[0,j].imshow(true, cmap=cmaps[0])
            axs[1,j].imshow(pred, cmap=cmaps[0])
            axs[2,j].imshow(np.abs(true-pred), cmap=cmaps[-1])
            axs[0,j].set_title('t={}'.format(self.t_samples[j]), weight='bold')
            for i in range(3):
                axs[i,j].set(xticks=[], yticks=[])
                axs[i,0].set_ylabel(labels[i], weight='bold')
        fig.suptitle('Pressure Results - R{} - {}'.format(realization, train_or_test), weight='bold'); plt.show()
        # Saturation results (dynamic output 1)
        fig, axs = plt.subplots(3, len(self.t_samples), figsize=figsize)
        for j in range(len(self.t_samples)):
            true, pred = y[realization,j,:,:,1], yhat[realization,j,:,:,1]
            axs[0,j].imshow(true, cmap=cmaps[0])
            axs[1,j].imshow(pred, cmap=cmaps[0])
            axs[2,j].imshow(np.abs(true-pred), cmap=cmaps[-1])
            axs[0,j].set_title('t={}'.format(self.t_samples[j]), weight='bold')
            for i in range(3):
                axs[i,j].set(xticks=[], yticks=[])
                axs[i,0].set_ylabel(labels[i], weight='bold')
        fig.suptitle('Saturation Results - R{} - {}'.format(realization, train_or_test), weight='bold'); plt.show()
        return None

    def cumulative_co2(self, threshold=0.02, figsize=(12,6)):
        def compare_cumulative_co2(true, pred, thresh=threshold):
            true_co2, pred_co2 = {}, {}
            for i in range(true.shape[0]):
                true_co2[i] = true[i][np.where(true[i]>thresh)].sum()
                pred_co2[i] = pred[i][np.where(pred[i]>thresh)].sum()
            true_co2 = np.array(list(true_co2.values()))
            pred_co2 = np.array(list(pred_co2.values()))
            return true_co2, pred_co2
        true_co2_train, pred_co2_train = compare_cumulative_co2(self.y_train, self.y_train_pred)
        true_co2_test,  pred_co2_test  = compare_cumulative_co2(self.y_test,  self.y_test_pred) 
        mean_train_true, mean_train_pred = true_co2_train.mean(), pred_co2_train.mean()
        mean_test_true,  mean_test_pred  = true_co2_test.mean(),  pred_co2_test.mean()
        print('Train - Mean CO2 Injected: True: {:.2f} | Pred: {:.2f}'.format(mean_train_true, mean_train_pred))
        print('Test  - Mean CO2 Injected: True: {:.2f} | Pred: {:.2f}'.format(mean_test_true,  mean_test_pred))
        fig = plt.figure(figsize=figsize)
        fig.suptitle('True vs. Predicted Cumulative CO$_2$ Injected', weight='bold')
        gs = GridSpec(2,2, hspace=0.25)
        ax1, ax2, ax3 = fig.add_subplot(gs[0,:]), fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1])
        ax1.axline([1000,1000],[10000,10000], c='r', linestyle='-', linewidth=3)
        ax1.scatter(true_co2_train, pred_co2_train, alpha=0.7, label='train')
        ax1.scatter(true_co2_test,  pred_co2_test,  alpha=0.7, label='test')
        ax1.set_xlabel('True', weight='bold'); ax1.set_ylabel('Predicted', weight='bold')
        ax1.legend(); ax1.grid('on'); 
        ax2.hist(true_co2_train, edgecolor='gray', label='true')
        ax2.hist(pred_co2_train, edgecolor='gray', alpha=0.75, label='pred')
        ax2.set_xlabel('Train', weight='bold'); ax2.legend()
        ax3.hist(true_co2_test, edgecolor='gray', label='true')
        ax3.hist(pred_co2_test, edgecolor='gray', alpha=0.75, label='pred')
        ax3.set_xlabel('Test', weight='bold'); ax3.legend()
        plt.show()
        return None
        
    def latent_space(self, train_or_test='train', plot=True, 
                        nrows=4, ncols=5, imult=200, jmult=30, alpha=0.65,
                        well_color='red', well_marker='$\swarrow$',
                        cmaps=['afmhot_r', 'gray', 'jet', 'jet'], figsize=(10,5)):
        if train_or_test == 'train':
            data = self.X_train
        elif train_or_test == 'test':
            data = self.X_test
        else:
            print('Please select "train" or "test" to display')
        self.z = self.encoder.predict(data)
        z_mean = self.z.mean(-1)
        print('Latent shape: {}'.format(self.z.shape))
        if self.return_data:
            return self.z
        titles = [r'Poro $\otimes$ $\overline{FM}$', r'Facies $\otimes$ $\overline{FM}$']
        if plot == True:
            _, axs = plt.subplots(nrows, ncols+2, figsize=figsize)
            for i in range(nrows):
                for j in range(ncols):
                    p, q = i*imult, j*jmult
                    poro, facies = data[p,:,:,0], data[p,:,:,2]
                    w = data[p,:,:,-1]; wlocs = np.argwhere(w!=0)
                    axs[i,j].imshow(self.z[p,:,:,q], cmap=cmaps[0])
                    axs[i,j].set(xticks=[], yticks=[])
                    axs[i,0].set_ylabel('R {}'.format(p)); axs[0,j].set_title('FM {}'.format(q))
                im1 = axs[i,ncols].imshow(poro, cmap=cmaps[2])
                axs[i,ncols].imshow(z_mean[i], cmap=cmaps[1], alpha=alpha, extent=im1.get_extent())
                axs[i,ncols].scatter(wlocs[:,1], wlocs[:,0], c=well_color, marker=well_marker)
                axs[0,ncols].set(title=titles[0]); axs[i,ncols].set(xticks=[], yticks=[])               
                im2 = axs[i,ncols+1].imshow(facies, cmap=cmaps[3])
                axs[i,ncols+1].imshow(z_mean[i], cmap=cmaps[1], alpha=alpha, extent=im2.get_extent())
                axs[i,ncols+1].scatter(wlocs[:,1], wlocs[:,0], c=well_color, marker=well_marker)
                axs[0,ncols+1].set(title=titles[1]); axs[i,ncols+1].set(xticks=[], yticks=[])
            plt.show()

    def feature_map_animation(self, nrows=4, ncols=8, imult=200, jmult=1, figsize=(15,5), cmap='afmhot_r',
                                blit=False, interval=750):
        z1 = self.latent1.predict(self.X_train)
        z2 = self.latent2.predict(self.X_train)
        z3 = self.encoder.predict(self.X_train)
        z = {0:z1, 1:z2, 2:z3}
        print('z1: {} | z2: {} | z3: {}'.format(z1.shape, z2.shape, z3.shape))
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
        for i in range(nrows):
            for j in range(ncols):
                p, q = i*imult, j*jmult
                axs[i,j].imshow(z[0][p,:,:,q], cmap=cmap)
                axs[i,j].set(xticks=[], yticks=[])
                axs[0,j].set(title='FM {}'.format(q))
            axs[i,0].set_ylabel('R {}'.format(p))
        def animate(k):
            for i in range(nrows):
                for j in range(ncols):
                    p, q = i*imult, j*jmult
                    axs[i,j].imshow(z[k][p,:,:,q], cmap=cmap)
                    axs[i,j].set(xticks=[], yticks=[])
                    axs[0,j].set(title='FM {}'.format(q))
                axs[i,0].set_ylabel('R {}'.format(p))
            return axs
        ani = FuncAnimation(fig, animate, frames=len(z), blit=blit, interval=interval)
        ani.save('figures/feature_maps_animation.gif')
        plt.show()
        return None

###################################################################################################
############################################### END ###############################################
###################################################################################################