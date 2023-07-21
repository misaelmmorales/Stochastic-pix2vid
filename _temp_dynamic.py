import os
import numpy as np


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
from timm.models.vision_transformer import SwinTransformer

class SwinTransformerModel(nn.Module):
    def __init__(self):
        super(SwinTransformerModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.patch_size = 4  # Swin Transformer patch size (adjust as needed)
        self.hidden_dim = 128  # Hidden dimension of the Swin Transformer

        self.swin_transformer = SwinTransformer(
            img_size=64,
            patch_size=self.patch_size,
            in_chans=32,
            num_classes=self.hidden_dim,
            embed_dim=128,
            depths=[2, 2, 6, 2],  # You can adjust the number of layers as needed
            num_heads=8,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.2,
            ape=False)

        self.output_layers = nn.Sequential(
            nn.Conv2d(self.hidden_dim, 2, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv_layers(x)
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, -1, self.patch_size, self.patch_size, 32)
        x = x.permute(0, 1, 4, 2, 3)
        x = x.view(batch_size, -1, self.patch_size * self.patch_size * 32)

        # Get the attention weights from the Swin Transformer
        attention_weights = []
        x, weights = self.swin_transformer(x, return_attention=True)
        attention_weights.append(weights)

        x = x.view(batch_size, -1, self.hidden_dim)
        x = x.permute(1, 0, 2)
        x = x.view(batch_size, 61, self.hidden_dim, self.patch_size, self.patch_size)
        x = F.interpolate(x, scale_factor=16)
        x = self.output_layers(x)
        return x, attention_weights