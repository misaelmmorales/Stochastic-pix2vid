import os
from time import time
from datetime import datetime
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from skimage.util import random_noise

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

class SpatiotemporalCO2:
    def __init__(self):
        K.clear_session()
        self.input_features_dir = 'simulations2D/input_features'
        self.output_targets_dir = 'simulations2D/output_targets'
        self.x_data_labels = ['Poro', 'LogPerm', 'Facies',  'Wells']
        self.y_data_labels = ['Pressure', 'Saturation']
        self.x_cmaps       = ['jet', 'jet', 'viridis', 'binary']
        self.y_cmaps       = ['jet', 'jet']
        self.err_cmap      = 'binary'
        self.latent_cmap   = ['afmhot_r', 'gray']
        self.return_data   = False
        self.save_model    = True

        self.n_samples   = 1000
        self.x_channels  = 4
        self.y_channels  = 2
        self.timesteps   = 60
        self.dim         = 64
        self.test_size   = 0.25
        self.t_samples   = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
        
        self.cnn_filters = [64,  128, 256]
        self.rnn_filters = [256, 128, 64]
        self.conv_groups = 1
        self.rnn_dropout = 0.05
        self.up_interp   = 'nearest'

        self.optimizer   = AdamW(learning_rate=1e-3, weight_decay=1e-5)
        self.criterion   = self.custom_loss
        self.L1L2_split  = 0.33
        self.ridge_alpha = 0.66
        self.regular     = regularizers.l1(1e-6)
        self.leaky_slope = 0.25
        self.valid_split = 0.20

        self.num_epochs  = 100
        self.batch_size  = 50
        self.lr_decay    = 20
        self.verbose     = 0

    ################################### MODEL ARCHITECTURE ####################################
    def encoder_layer(self, inp, filt, kern=3, pad='same'):
        _ = SeparableConv2D(filt, kern, padding=pad, activity_regularizer=self.regular)(inp)
        _ = SqueezeExcite()(_)
        _ = InstanceNormalization()(_)
        _ = PReLU()(_)
        _ = MaxPooling2D()(_)
        _ = SpatialDropout2D(self.rnn_dropout)(_)
        return _
    
    def recurrent_decoder(self, z_input, residuals, previous_timestep=None):
        dropout = self.rnn_dropout
        def recurrent_step(inp, filt, res, kern=3, pad='same', drop=dropout):
            y = ConvLSTM2D(filt, kern, padding=pad)(inp)
            y = BatchNormalization()(y)
            y = LeakyReLU(self.leaky_slope)(y)
            y = Conv2DTranspose(filt, kern, padding=pad, strides=2)(y)
            y = SpatialDropout2D(drop)(y)
            y = Concatenate()([y, res])
            y = Conv2D(filt, kern, padding=pad)(y)
            y = Activation('sigmoid')(y)
            y = tf.expand_dims(y,1)
            return y
        def recurrent_last(inp, filt, kern=3, pad='same', drop=dropout):
            y = ConvLSTM2D(filt, kern, padding=pad)(inp)
            y = BatchNormalization()(y)
            y = LeakyReLU(self.leaky_slope)(y)
            y = Conv2DTranspose(filt, kern, padding=pad, strides=2)(y)
            y = SpatialDropout2D(drop)(y)
            y = Conv2D(self.y_channels, kern, padding=pad)(y)
            y = Activation('sigmoid')(y)
            y = tf.expand_dims(y, 1)
            return y
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
        if self.verbose != 0:
            print('# Parameters: {:,}'.format(self.model.count_params()))
        if self.return_data:
            return self.model

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

    def training(self):
        self.model.compile(optimizer=self.optimizer, loss=self.criterion, metrics=['mse'])
        loss_cb   = self.LossCallback()
        lr_reduce = ReduceLROnPlateau(patience=self.lr_decay) 
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
        print('# Parameters: {:,} | Batch size: {}  | Epochs: {}'.format(n_params, self.batch_size, self.num_epochs))
        start = time()
        self.fit = self.model.fit(self.X_train, self.y_train,
                                    epochs           = self.num_epochs,
                                    batch_size       = self.batch_size,
                                    validation_split = self.valid_split,
                                    shuffle          = True,
                                    callbacks        = cb,
                                    verbose          = self.verbose)
        train_time = time()-start
        print('Training Time: {:.2f} minutes'.format(train_time/60))
        self.plot_loss()
        if self.save_model:
            self.model.save('stochastic_pix2vid.keras')
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

    ####################################### DATA LOADING ######################################
    def load_data(self):
        print('... Loading Full Dataset ...')
        X_data = np.zeros((self.n_samples, self.x_channels, self.dim, self.dim))
        y_data = np.zeros((self.n_samples, self.timesteps, self.y_channels, self.dim, self.dim))
        for i in range(self.n_samples):
            X_data[i] = np.load(self.input_features_dir + '/X_data_{}.npy'.format(i))
            y_data[i] = np.load(self.output_targets_dir + '/y_data_{}.npy'.format(i))
        self.X_data, self.y_data = np.moveaxis(X_data, 1, -1), np.moveaxis(y_data, 2, -1)
        print('X: {} | y: {}'.format(self.X_data.shape, self.y_data.shape))
    
    def apply_noise(self, image_array, noise_type='gaussian', var=1e-6):
        noisy_array = np.zeros_like(image_array)
        for i in range(image_array.shape[0]):
            for c in range(image_array.shape[-1]-1):
                if (noise_type=='gaussian' or noise_type=='speckle'):
                    noisy_array[i,:,:,c] = random_noise(image_array[i,:,:,c], mode=noise_type, var=var)                    
                else:
                    noisy_array[i,:,:,c] = random_noise(image_array[i,:,:,c], mode=noise_type)
            noisy_array[...,-1] = image_array[...,-1] #keep Wells channel same
        return noisy_array
        
    def process_data(self, n_subsample=None, augment=True, rotations=3, add_noise=False):
        # data augmentation
        if augment==True:
            xrot = np.rot90(self.X_data, k=rotations, axes=(1,2))
            yrot = np.rot90(self.y_data, k=rotations, axes=(2,3))
            x = np.concatenate([self.X_data, xrot], axis=0)
            y = np.concatenate([self.y_data, yrot], axis=0)
            print('Data Augmentation Done!    - n_samples={:,}'.format(x.shape[0]))
        else:
            x = self.X_data
            y = self.y_data
        # feature processing
        num, height, width, channels = x.shape
        xn = MinMaxScaler().fit_transform(x.reshape(num*height*width, channels)).reshape(x.shape)
        if add_noise == True:
            self.X_norm = self.apply_noise(xn)
        else:
            self.X_norm = xn
        # target processing
        num, tsteps, height, width, channels = y.shape
        self.y_norm = MinMaxScaler().fit_transform(y.reshape(num*tsteps, -1)).reshape(y.shape)
        print('MinMax Normalization Done! - [{}, {}]'.format(self.X_norm.min(), self.X_norm.max()))
        # train-test split and subsampling
        ts = np.array(self.t_samples); ts[1:]-=1
        if n_subsample != None:
            print('\n... Subsampling data for {} samples, {} timesteps ...'.format(n_subsample, len(self.t_samples)))
            idx = np.random.choice(range(self.X_norm.shape[0]), n_subsample, replace=False)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_norm[idx], self.y_norm[idx][:,ts], test_size=self.test_size)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_norm, self.y_norm[:,ts], test_size=self.test_size)
        if self.save_model == True:
            np.savez('xy_data.npz', X_train=self.X_train, X_test=self.X_test, y_train=self.y_train, y_test=self.y_test)
        print('Train - X: {} | y: {}'.format(self.X_train.shape, self.y_train.shape))
        print('Test  - X: {}  | y: {}'.format(self.X_test.shape, self.y_test.shape))
        
    def load_preprocessed_xy_train_test(self, filename):
        loaded_data = np.load(filename)
        self.X_train, self.X_test = loaded_data['X_train'], loaded_data['X_test']
        self.y_train, self.y_test = loaded_data['y_train'], loaded_data['y_test']
        return (self.X_train, self.X_test), (self.y_train, self.y_test)

    ########################################## PLOTS ##########################################
    def plot_loss(self, figsize=(5,3), cs=['tab:blue','tab:orange'], grid=True, lines=['-','--']):
        ep = len(self.fit.history['loss'])
        it = np.arange(ep)
        plt.figure(figsize=figsize)
        plt.plot(it, self.fit.history['loss'], c=cs[0], linestyle=lines[0], label='loss')
        plt.plot(it, self.fit.history['val_loss'], c=cs[1], linestyle=lines[1], label='validation loss')
        plt.title('Training: Loss vs. Epochs'); plt.xlabel('Epochs'); plt.ylabel('Loss')
        plt.legend(); plt.xticks(it[::ep//10])
        if grid==True:
            plt.grid('on')
        plt.show()
            
    def plot_features(self, nsamples=10, multiplier=1, figsize=(15,5)):
        _, axs = plt.subplots(4, nsamples, figsize=figsize)
        for i in range(4):
            for j in range(nsamples):
                axs[i,j].imshow(self.X_norm[j*multiplier,:,:,i], cmap=self.x_cmaps[i])
                axs[i,j].set(xticks=[], yticks=[])
                axs[i,0].set_ylabel(self.x_data_labels[i], weight='bold')
                axs[0,j].set_title('R{}'.format(j*multiplier), weight='bold')
        plt.show()
    
    def plot_targets(self, p_s=1, nsamples=10, multiplier=1, masked=False, figsize=(15,10)):
        ts = np.array(self.t_samples); ts[1:]-=1
        _, axs = plt.subplots(len(ts), nsamples, figsize=figsize)
        for i in range(len(ts)):
            for j in range(nsamples):
                yy = self.y_norm[j*multiplier, ts[i], :, :, p_s]
                if masked==True:
                    yy = np.ma.masked_where(yy==0, yy)
                axs[i,j].imshow(yy, cmap=self.y_cmaps[p_s])
                axs[i,j].set(xticks=[], yticks=[])
                axs[0,j].set_title('R{}'.format(j*multiplier), weight='bold')
                axs[i,0].set_ylabel('t={}'.format(ts[i]+1), weight='bold')
        plt.show()
    
    def plot_data(self, nsamples=10, multiplier=200, p_s=1, masked=False, figsize=(12,15)):
        ts = np.array(self.t_samples); ts[1:]-=1
        nx, ny = self.x_channels, len(ts)
        _, axs = plt.subplots(nx+ny, nsamples, figsize=figsize, tight_layout=True)
        # features
        for i in range(nx):
            for j in range(nsamples):
                xx = self.X_norm[j*multiplier, :, :, i]
                axs[i,j].imshow(xx, cmap=self.x_cmaps[i])
            axs[i,0].set_ylabel(self.x_data_labels[i], weight='bold')
        # targets
        for i in range(ny):
            for j in range(nsamples):
                p = i+nx
                yy = self.y_norm[j*multiplier, ts[i], :, :, p_s]
                if masked==True:
                    yy = np.ma.masked_where(yy==0, yy)
                axs[p,j].imshow(yy, cmap=self.y_cmaps[p_s])
            axs[p,0].set_ylabel('t={}'.format(ts[i]+1), weight='bold')
        # plotting
        for j in range(nsamples):
            axs[0,j].set_title('R{}'.format(j*multiplier), weight='bold')
            for i in range(nx+ny):
                axs[i,j].set(xticks=[], yticks=[])
        plt.show()

    def plot_single_results(self, realization, train_or_test='train', masked=False, figsize=(20,4)):
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
            plt.imshow(x[realization,:,:,i], cmap=self.x_cmaps[i])
            plt.xticks([]); plt.yticks([]); plt.title(self.x_data_labels[i], weight='bold')
        plt.suptitle('Geologic Models - R{}'.format(realization), weight='bold'); plt.show()
        # Reservoir response (dynamic outputs)
        for k in range(2):
            fig, axs = plt.subplots(3, len(self.t_samples), figsize=figsize)
            for j in range(len(self.t_samples)):
                true, pred = y[realization, j, :, :, k], yhat[realization, j, :, :, k]                  
                if masked==True:
                    true = np.ma.masked_where(true==0, true)
                    pred = np.ma.masked_where(pred==0, pred)
                axs[0,j].imshow(true, cmap=self.y_cmaps[k])
                axs[1,j].imshow(pred, cmap=self.y_cmaps[k])
                axs[2,j].imshow(np.abs(true-pred), cmap=self.err_cmap)
                axs[0,j].set_title('t={}'.format(self.t_samples[j]), weight='bold')
                for i in range(3):
                    axs[i,j].set(xticks=[], yticks=[])
                    axs[i,0].set_ylabel(labels[i], weight='bold')
            fig.suptitle(self.y_data_labels[k] + ' Results - R{} - {}'.format(realization, train_or_test), weight='bold')
            plt.show()

    def latent_space(self, train_or_test='train', plot=True, 
                     nrows=4, ncols=5, imult=1, jmult=None, alpha=0.65,
                     well_color='red', well_marker='$\swarrow$', figsize=(10,5)):
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
            if jmult == None:
                jmult = self.z.shape[-1]//ncols
            _, axs = plt.subplots(nrows, ncols+2, figsize=figsize)
            for i in range(nrows):
                for j in range(ncols):
                    p, q = i*imult, j*jmult
                    poro, facies = data[p,:,:,0], data[p,:,:,2]
                    w = data[p,:,:,-1]; wlocs = np.argwhere(w!=0)
                    axs[i,j].imshow(self.z[p,:,:,q], cmap=self.latent_cmap[0])
                    axs[i,j].set(xticks=[], yticks=[])
                    axs[i,0].set_ylabel('R {}'.format(p), weight='bold')
                    axs[0,j].set_title('FM {}'.format(q), weight='bold')
                im1 = axs[i,ncols].imshow(poro, cmap=self.x_cmaps[0])
                axs[i,ncols].imshow(z_mean[i], cmap=self.latent_cmap[1], alpha=alpha, extent=im1.get_extent())
                axs[i,ncols].scatter(wlocs[:,1], wlocs[:,0], c=well_color, marker=well_marker)
                axs[0,ncols].set(title=titles[0]); axs[i,ncols].set(xticks=[], yticks=[])               
                im2 = axs[i,ncols+1].imshow(facies, cmap=self.x_cmaps[2])
                axs[i,ncols+1].imshow(z_mean[i], cmap=self.latent_cmap[1], alpha=alpha, extent=im2.get_extent())
                axs[i,ncols+1].scatter(wlocs[:,1], wlocs[:,0], c=well_color, marker=well_marker)
                axs[0,ncols+1].set(title=titles[1]); axs[i,ncols+1].set(xticks=[], yticks=[])
            plt.show()

    def feature_map_animation(self, nrows=4, ncols=8, imult=200, jmult=1, figsize=(15,5), blit=False, interval=900):
        z1 = self.latent1.predict(self.X_train)
        z2 = self.latent2.predict(self.X_train)
        z3 = self.encoder.predict(self.X_train)
        z = {0:z1, 1:z2, 2:z3}
        print('z1: {} | z2: {} | z3: {}'.format(z1.shape, z2.shape, z3.shape))
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
        for i in range(nrows):
            for j in range(ncols):
                p, q = i*imult, j*jmult
                axs[i,j].imshow(z[0][p,:,:,q], cmap=self.latent_cmap[0])
                axs[i,j].set(xticks=[], yticks=[])
                axs[0,j].set(title='FM {}'.format(q))
            axs[i,0].set_ylabel('R {}'.format(p))
        def animate(k):
            for i in range(nrows):
                for j in range(ncols):
                    p, q = i*imult, j*jmult
                    axs[i,j].imshow(z[k][p,:,:,q], cmap=self.latent_cmap[0])
                    axs[i,j].set(xticks=[], yticks=[])
                    axs[0,j].set(title='FM {}'.format(q))
                axs[i,0].set_ylabel('R {}'.format(p))
            return axs
        ani = FuncAnimation(fig, animate, frames=len(z), blit=blit, interval=interval)
        ani.save('figures/feature_maps_animation.gif')
        plt.show()
        return None
    
    def cumulative_co2(self, threshold=0.02, alpha=0.7, figsize=(12,6)):
        def compare_cumulative_co2(true, pred, thresh=threshold):
            true_co2, pred_co2 = {}, {}
            for i in range(true.shape[0]):
                true_co2[i] = true[i][np.where(true[i]>thresh)].sum()
                pred_co2[i] = pred[i][np.where(pred[i]>thresh)].sum()
            true_co2 = np.array(list(true_co2.values()))
            pred_co2 = np.array(list(pred_co2.values()))
            return true_co2, pred_co2
        true_co2_train,  pred_co2_train  = compare_cumulative_co2(self.y_train, self.y_train_pred)
        true_co2_test,   pred_co2_test   = compare_cumulative_co2(self.y_test,  self.y_test_pred) 
        mean_train_true, mean_train_pred = true_co2_train.mean(), pred_co2_train.mean()
        mean_test_true,  mean_test_pred  = true_co2_test.mean(),  pred_co2_test.mean()
        print('Train - Mean CO2 Injected: True: {:.2f} | Pred: {:.2f}'.format(mean_train_true, mean_train_pred))
        print('Test  - Mean CO2 Injected: True: {:.2f} | Pred: {:.2f}'.format(mean_test_true,  mean_test_pred))
        fig = plt.figure(figsize=figsize)
        fig.suptitle('True vs. Predicted Cumulative CO$_2$ Injected', weight='bold')
        gs = GridSpec(2,2, hspace=0.25)
        ax1, ax2, ax3 = fig.add_subplot(gs[0,:]), fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1])
        ax1.axline([1000,1000],[10000,10000], c='r', linestyle='-', linewidth=3)
        ax1.scatter(true_co2_train, pred_co2_train, alpha=alpha, label='train')
        ax1.scatter(true_co2_test,  pred_co2_test,  alpha=alpha, label='test')
        ax1.set_xlabel('True', weight='bold'); ax1.set_ylabel('Predicted', weight='bold')
        ax1.legend(); ax1.grid('on'); 
        ax2.hist(true_co2_train, edgecolor='gray', label='true')
        ax2.hist(pred_co2_train, edgecolor='gray', alpha=alpha, label='pred')
        ax2.set_xlabel('Train', weight='bold'); ax2.legend()
        ax3.hist(true_co2_test, edgecolor='gray', label='true')
        ax3.hist(pred_co2_test, edgecolor='gray', alpha=alpha, label='pred')
        ax3.set_xlabel('Test', weight='bold'); ax3.legend()
        plt.show()
        return None

############################################ UTILITIES ############################################
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
    print('--------------------------------------')
    return None

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
    
############################################## MAIN ###############################################
if __name__ == '__main__':
    print('-'*80+'\n'+'Date:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("Current Working Directory:", os.getcwd())
    
    check_tensorflow_gpu()
    proxy = SpatiotemporalCO2()
    proxy.__dict__

    proxy.load_data()
    proxy.process_data()

    proxy.plot_data()
    proxy.plot_features()
    proxy.plot_targets()

    proxy.make_model()
    proxy.training()
    proxy.predictions()
    proxy.plot_single_results(411, 'train')
    proxy.cumulative_co2()
    
###################################################################################################
############################################### END ###############################################
###################################################################################################