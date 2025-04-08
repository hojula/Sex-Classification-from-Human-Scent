import torch.nn as nn
from einops import rearrange

import torch
import torch.nn.functional as F
from einops import rearrange
import cv2
import numpy as np
import os


class MyModel(nn.Module):
    # Defeniton of FCN network used for the compound detection
    def __init__(self, num_classes, num_conv_layers):
        super(MyModel, self).__init__()
        self.convs_spectra = nn.ModuleList()
        self.convs_times = nn.ModuleList()
        self.batch_norms_spectra = nn.ModuleList()
        self.batch_norms_times = nn.ModuleList()
        self.relu = nn.ReLU()
        self.in_channels_times = 2
        for i in range(num_conv_layers):
            self.out_channels_times = min(32 * 2 ** i, 32 * 2 ** 6)
            kernel_size = 17
            self.convs_times.append(
                nn.Conv1d(in_channels=self.in_channels_times, out_channels=self.out_channels_times,
                          kernel_size=kernel_size,
                          padding=1,
                          stride=2)
            )
            self.batch_norms_times.append(nn.BatchNorm1d(self.out_channels_times))
            print("Conv times", self.in_channels_times, self.out_channels_times)
            self.in_channels_times = self.out_channels_times

        self.in_channels_spectra = 1
        for i in range(num_conv_layers):
            self.out_channels_conv = min(32 * 2 ** i, 32 * 2 ** 6)
            kernel_size = 17
            if i == 0:
                self.convs_spectra.append(
                    nn.Conv1d(in_channels=self.in_channels_spectra, out_channels=self.out_channels_conv,
                              kernel_size=kernel_size,
                              padding=1,
                              stride=2)
                )
            else:
                print("in_channels", self.in_channels_spectra)
                self.in_channels_spectra += self.convs_times[i - 1].out_channels
                print("in_channels after", self.in_channels_spectra)
                self.convs_spectra.append(
                    nn.Conv1d(in_channels=self.in_channels_spectra, out_channels=self.out_channels_conv,
                              kernel_size=kernel_size,
                              padding=1,
                              stride=2)
                )
            self.batch_norms_spectra.append(nn.BatchNorm1d(self.out_channels_conv))
            print("Conv spectra", self.in_channels_spectra, self.out_channels_conv)
            self.in_channels_spectra = self.out_channels_conv
        # 98 for 2 conv 4 stride
        self.last_conv_spectra = nn.Conv1d(2 * self.in_channels_spectra, 1, kernel_size=14, stride=1, padding=0)
        # self.last_conv_spectra = nn.Conv1d(2 * self.in_channels_spectra, 1, kernel_size=9)
        self.last_conv_times = nn.Conv1d(self.in_channels_times, 1, kernel_size=8)

        # Fully connected layer with layer norm and dropout
        # 80 for 2 conv and stride 3
        # 39 for 4 conv and stride 2 + 2
        # 1 for 6 conv and stride 2

    def forward(self, x):
        spectra, x_coord, y_coord = x[:, 0:1, :, :], x[:, 1, 0, 0], x[:, 2, 0, 0]
        times = x[:, 1:, :, :]
        spectra = rearrange(spectra, 'b c h w -> b c (h w)')
        times = rearrange(times, 'b c h w -> b c (h w)')
        # print(spectra.shape)
        # print(times.shape)
        i = 0
        for conv_spectra, conv_times in zip(self.convs_spectra, self.convs_times):
            spectra = conv_spectra(spectra)  # Apply convolution
            spectra = self.batch_norms_spectra[i](spectra)
            spectra = self.relu(spectra)  # Apply ReLU activation
            times = conv_times(times)  # Apply convolution
            times = self.batch_norms_times[i](times)
            times = self.relu(times)  # Apply ReLU activation
            spectra = torch.cat((spectra, times), dim=1)
            i += 1

        spectra = self.last_conv_spectra(spectra)
        spectra = rearrange(spectra, 'b c h -> b (c h)')
        output = spectra

        return output
