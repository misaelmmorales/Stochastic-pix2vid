import os, sys
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

from neuralop.models import TFNO
from neuralop.utils import count_params
from neuralop import LpLoss, H1Loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import VisionTransformer
from torchmetrics.image import StructuralSimilarityIndexMeasure as torch_SSIM

class torch_SpatioTemporalCO2:
    def __init__(self):
        self.input_features_dir = 'simulations2D/input_features'
        self.X_filenames        = os.listdir(self.input_features_dir)
        self.output_targets_dir = 'simulations2D/output_targets'
        self.y_filenames        =  os.listdir(self.output_targets_dir)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.n_realizations = 1000
        self.x_channels  = 4
        self.y_channels  = 2
        self.timesteps   = 60
        self.dim         = 64
        self.test_size   = 0.25

        self.batch_size = 32

    def check_torch_gpu(self):
        '''
        Check torch build in python to ensure GPU is available for training.
        '''
        torch_version, cuda_avail = torch.__version__, torch.cuda.is_available()
        count, name = torch.cuda.device_count(), torch.cuda.get_device_name()
        py_version, conda_env_name = sys.version, sys.executable.split('\\')[-2]
        print('-------------------------------------------------')
        print('------------------ VERSION INFO -----------------')
        print('Conda Environment: {}'.format(conda_env_name))
        print('Torch version: {}'.format(torch_version))
        print('Torch build with CUDA? {}'.format(cuda_avail))
        print('# Device(s) available: {}, Name(s): {}\n'.format(count, name))
        self.device = torch.device('cuda' if cuda_avail else 'cpu')
        return self.device

    # Define custom dataset classes for train and test sets
    class CustomDataset(Dataset):
        def __init__(self, filenames):
            self.root_X = torch_SpatioTemporalCO2().input_features_dir
            self.root_y = torch_SpatioTemporalCO2().output_targets_dir
            self.filenames_X, self.filenames_y = filenames
        def __len__(self):
            return len(self.filenames_X)
        def __getitem__(self, idx):
            X_path = os.path.join(self.root_X, self.filenames_X[idx])
            y_path = os.path.join(self.root_y, self.filenames_y[idx])
            X_array, y_array = np.load(X_path), np.load(y_path)
            X_normalized = (X_array - X_array.min()) / (X_array.max() - X_array.min())
            y_normalized = (y_array - y_array.min()) / (y_array.max() - y_array.min())
            X_tensor = torch.from_numpy(X_normalized)
            y_tensor = torch.from_numpy(y_normalized)
            return X_tensor, y_tensor

    def make_dataloaders(self):
        X_train_fname, X_test_fname, y_train_fname, y_test_fname = train_test_split(self.X_filenames, self.y_filenames, test_size=self.test_size)
        self.train_dataset = self.CustomDataset([X_train_fname, y_train_fname])
        self.test_dataset  = self.CustomDataset([X_test_fname,  y_test_fname])
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader  = DataLoader(self.test_dataset,  batch_size=self.batch_size, shuffle=True)
        return self.train_dataloader, self.test_dataloader, self.train_dataset, self.test_dataset
    
class encoder_layer(nn.Module):
    def __init__(self, in_ch, out_ch, hidden, n_modes=(16,16), lifting=256, projection=256, n_layers=4, separable=True, leaky_slope=0.2):
        super(encoder_layer, self).__init__()
        self.fno   = TFNO(n_modes, hidden, in_ch, out_ch, lifting, projection, n_layers, separable=True)
        self.inorm = nn.InstanceNorm2d(out_ch)
        self.activ = nn.LeakyReLU(leaky_slope)
        self.bnorm = nn.BatchNorm2d(out_ch)
        self.pool  = nn.MaxPool2d(2,2)
    def forward(self, x):
        x = self.fno(x)
        x = self.inorm(x)
        x = self.activ(x)
        x = self.bnorm(x)
        x = self.pool(x)
        return x

class decoder_layer(nn.Module):
    def __init__(self, in_ch, out_ch, leaky_slope=0.2):
        super(decoder_layer, self).__init__()
        self.space_conv = nn.ConvTranspose3d(in_ch, out_ch, (1,2,2), stride=(1,2,2))
        #self.time_conv  = TFNO((16,16), (in_ch+out_ch)//2, in_ch, out_ch)
        self.activ      = nn.LeakyReLU(leaky_slope)
        self.bnorm      = nn.BatchNorm3d(out_ch)
    def forward(self, x):
        x = self.space_conv(x)
        #x = self.time_conv(x)
        x = self.activ(x)
        x = self.bnorm(x)
        return x

class recurrent_layer(nn.Module):
    def __init__(self, repeats):
        super(recurrent_layer, self).__init__()
        self.repeats = repeats
        self.lstm1 = nn.LSTM(batch_first=True, input_size=self.repeats, hidden_size=60)
        self.lstm2 = nn.LSTM(batch_first=True, input_size=60, hidden_size=60)
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, channels*height*width, 1).repeat(1, 1, self.repeats)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = torch.moveaxis(x, -1, 1)
        x = x.view(batch_size, 60, channels, height, width)
        x = torch.moveaxis(x, 1, 2)
        return x

class ProxyModel(nn.Module):
    def __init__(self):
        super(ProxyModel, self).__init__()
        self.encoder = nn.Sequential(
            encoder_layer(4,  16,  hidden=8),
            encoder_layer(16, 64,  hidden=32),
            encoder_layer(64, 128, hidden=96))
        self.recurrence = recurrent_layer(30)
        self.decoder = nn.Sequential(
            decoder_layer(128, 64),
            decoder_layer(64, 32),
            decoder_layer(32, 16))
        self.out_layer = nn.Sequential(
            nn.Conv3d(16, 2, kernel_size=3, padding=1),
            nn.Sigmoid())
    def forward(self, x):
        x = self.encoder(x)
        x = self.recurrence(x)
        x = self.decoder(x)
        x = self.out_layer(x)
        x = torch.moveaxis(x, 2, 1)
        return x


class CustomLoss(nn.Module):
    def __init__(self, mse_weight=(2/3), ridge_weight=0.8):
        super(CustomLoss, self).__init__()
        self.mse_weight = mse_weight
        self.ridge_weight = ridge_weight
        self.ssim = torch_SSIM(data_range=1.0)

    def forward(self, output, target):
        # ridge loss
        mse_loss = F.mse_loss(output, target)
        mae_loss = F.l1_loss(output, target)
        ridge_loss = self.mse_weight * mse_loss + (1-self.mse_weight) * mae_loss
        # ssim loss
        ssim_losses = []
        for t in range(output.size(1)):
                pred_single = output[:, t]
                target_single = target[:, t]
                ssim_loss = 1 - self.ssim(pred_single, target_single)
                ssim_losses.append(ssim_loss)
        ssim_loss = torch.stack(ssim_losses).mean()
        # total loss
        loss = self.ridge_weight * ridge_loss + (1-self.ridge_weight) * ssim_loss
        return loss


''' 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProxyModel()
model.to(device)

criterion  = CustomLoss().to(device)
optimizer  = torch.optim.NAdam(model.parameters(), lr=0.001)
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    train_subset_size = int(len(train_dataset) * 0.8)  # 80% for training
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_subset_size, len(train_dataset) - train_subset_size])
    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    for batch_idx, (x, y) in enumerate(train_dataloader):
        x, y = x.float().to(device), y.float().to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    tot_train_loss = train_loss/len(train_dataloader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(valid_dataloader):
            x, y = x.float().to(device), y.float().to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            val_loss += loss.item()
    tot_valid_loss = val_loss/len(valid_dataloader)
    
    if (epoch+1) % 5 == 0:
        print('Epoch: [{}/{}] | Loss: {:.4f} | Validation Loss: {:.4f}'.format(epoch+1, num_epochs, tot_train_loss, tot_valid_loss))

print("Training finished.")
'''