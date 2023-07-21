import os, sys, glob, math, re, cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyvista as pv
from time import time

from cv2 import resize
from scipy.stats import zscore
from scipy.io import loadmat, savemat
from numpy.matlib import repmat
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchvision.utils import save_image
from torchsummary import summary
from torchviz import make_dot
import torchio as tio


def check_torch_gpu(self):
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
    self.device = torch.device('cuda' if cuda_avail else 'cpu')
    return None
    
class Generator(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Generator, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        # Latent Transformer
        #self.latent_transformer = SwinTransformer(512, output_channels)
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh())
    def forward(self, x):
        encoded = self.encoder(x)
        #latent  = self.latent_transformer(encoded)
        #decoded = self.decoder(latent)
        decoded = self.decoder(encoded)
        return decoded
    
class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1))
    def forward(self, x):
        out = self.model(x)
        return out

class CycleGAN(nn.Module):
    def __init__(self, input_channels_X, output_channels_Y, input_channels_Y, output_channels_X):
        super(CycleGAN, self).__init__()
        self.generator_XY    = Generator(input_channels_X, output_channels_Y)
        self.generator_YX    = Generator(input_channels_Y, output_channels_X)
        self.discriminator_X = Discriminator(input_channels_X)
        self.discriminator_Y = Discriminator(input_channels_Y)
    
    def forward(self, x, y):
        fake_X = self.generator_YX(y)
        fake_Y = self.generator_XY(x)
        reconstructed_X = self.generator_YX(fake_Y)
        reconstructed_Y = self.generator_XY(fake_X)
        return fake_X, fake_Y, reconstructed_X, reconstructed_Y
    
    def train(self, dataloader, num_epochs, optimizer_list, loss_list, monitor=10, device='cuda', return_data=True, verbose=True, save=False):
        optimizer_G, optimizer_D_X, optimizer_D_Y = optimizer_list
        cycle_consistency_loss, adversarial_loss  = loss_list
        for epoch in range(num_epochs):
            for i, (X,Y) in enumerate(dataloader):
                X = X.to(device)
                Y = Y.to(device)
                # Adversarial ground truths
                valid = torch.ones(X.size(0),  1, 6, 6).to(device)
                fake  = torch.zeros(X.size(0), 1, 6, 6).to(device)
                # ------------------
                #  Train Generators
                # ------------------
                optimizer_G.zero_grad()
                # Identity loss
                identity_X = self.generator_YX(Y)
                identity_Y = self.generator_XY(X)
                loss_identity = cycle_consistency_loss(identity_X, X) + cycle_consistency_loss(identity_Y, Y)
                # Adversarial loss
                fake_X, fake_Y, reconstructed_X, reconstructed_Y = self.forward(X,Y)
                loss_GAN_XY = adversarial_loss(self.discriminator_Y(fake_Y), valid)
                loss_GAN_YX = adversarial_loss(self.discriminator_X(fake_X), valid)
                loss_GAN = loss_GAN_XY + loss_GAN_YX
                # Cycle Consistency loss
                loss_cycle_X = cycle_consistency_loss(reconstructed_X, X)
                loss_cycle_Y = cycle_consistency_loss(reconstructed_Y, Y)
                loss_cycle = loss_cycle_X + loss_cycle_Y
                # Total generator loss
                loss_G = loss_identity + loss_GAN + 10*loss_cycle
                loss_G.backward()
                optimizer_G.step()
                # ---------------------
                #  Train Discriminators
                # ---------------------
                # Discriminator X loss
                optimizer_D_X.zero_grad()
                loss_real = adversarial_loss(self.discriminator_X(X), valid)
                loss_fake = adversarial_loss(self.discriminator_X(fake_X.detach()), fake)
                loss_D_X = (loss_real + loss_fake)/2
                loss_D_X.backward()
                optimizer_D_X.step()
                # Discriminator Y loss
                optimizer_D_Y.zero_grad()
                loss_real = adversarial_loss(self.discriminator_Y(Y), valid)
                loss_fake = adversarial_loss(self.discriminator_Y(fake_Y.detach()), fake)
                loss_D_Y = (loss_real + loss_fake)/2
                loss_D_Y.backward()
                optimizer_D_Y.step()
                # Loss Dictionary
                final_loss = {'loss_G':loss_G.item(), 'loss_D_X':loss_D_X.item(), 'loss_D_Y':loss_D_Y.item()}
            if (epoch+1) % monitor == 0:
                if verbose:
                    print('Epoch [{}/{}]: Generator Loss: {:.3f}, Discriminator Loss: {:.3f}'.format(
                        epoch+1, num_epochs, loss_G.item(), loss_D_X.item()+loss_D_Y.item()))
                with torch.no_grad():
                    fake_X, fake_Y, reconstructed_X, reconstructed_Y = self.forward(X,Y)
                    fake_images = torch.cat([Y, fake_Y], dim=0)
                if return_data:
                    return fake_images, final_loss
                if save:
                    torch.save(fake_images, 'generated_images/epoch_{}.png'.format(epoch+1))

        if save:
            torch.save(self.generator_XY.state_dict(),    'generator_XY.pt')
            torch.save(self.generator_YX.state_dict(),    'generator_YX.pt')
            torch.save(self.discriminator_X.state_dict(), 'discriminator_X.pt')
            torch.save(self.discriminator_Y.state_dict(), 'discriminator_Y.pt')
            
            
class NumpyDataset_fromFolder(Dataset):
    def __init__(self, folder_X, folder_y):
        self.folder_X = folder_X
        self.folder_y = folder_y
        self.X_filenames = sorted(os.listdir(folder_X))
        self.y_filenames = sorted(os.listdir(folder_y))
    def __len__(self):
        return len(self.X_filenames)
    def __getitem__(self, index):
        X_path = os.path.join(self.folder_X, self.X_filenames[index])
        y_path = os.path.join(self.folder_y, self.y_filenames[index])
        X, y = np.load(X_path), np.load(y_path)
        return X, y
    
class NumpyDataset_from_array(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        img_x = self.X[index]
        img_y = self.y[index]        
        return img_x, img_y
    
class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(PatchMerging, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.LayerNorm(out_channels)
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.gelu(x)
        x = x.permute(0, 2, 3, 1)  # B, H, W, C
        B, H, W, C = x.shape
        x = x.reshape(B, H // 2, 2, W // 2, 2, C)
        x = x.permute(0, 1, 3, 2, 4, 5)  # B, H // 2, W // 2, 2, 2, C
        x = x.reshape(B, H // 2, W // 2, -1)
        x = x.permute(0, 3, 1, 2)  # B, 4C, H // 2, W // 2
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=4):
        super(PatchEmbedding, self).__init__()
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # B, H, W, C
        return x

class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4):
        super(SwinBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim))
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = x.permute(1, 0, 2)  # H, B, C
        x, _ = self.attn(x, x, x)
        x = x.permute(1, 0, 2)  # B, H, C
        x = x + residual
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        return x

class SwinTransformer(nn.Module):
    def __init__(self, in_channels, out_channels,
                 img_size=64, patch_size=4, embed_dim=96,
                 depths=[2, 2, 6, 2], num_heads=3, mlp_ratio=4):
        super(SwinTransformer, self).__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.merging_blocks = nn.ModuleList([PatchMerging(embed_dim, embed_dim*2, stride=2) for _ in range(len(depths))])
        self.blocks = nn.ModuleList([SwinBlock(embed_dim*4, num_heads, mlp_ratio=mlp_ratio) for _ in range(sum(depths))])
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim*4), nn.Linear(embed_dim*4, out_channels))
    def forward(self, x):
        x = self.patch_embed(x)
        B, N, C = x.shape
        x = x + self.pos_embed
        for merging_block, block in zip(self.merging_blocks, self.blocks):
            x = merging_block(x)
            x = block(x)
        x = x.mean(dim=1)
        x = self.mlp_head(x)
        return x