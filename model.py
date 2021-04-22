import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from dn3.trainable.models import EEGNet as DN3EEGNet, TIDNet
from adaptive_normalization import AdaptiveInputNorm, DAIN_Layer


def get_padding_sequence(channels, timepoints, f_height, f_width) -> Tuple[int, int, int, int]:
    strides = (None, 1, 1)

    # The total padding applied along the height and width is computed as:

    if channels % strides[1] == 0:
        pad_along_height = max(f_height - strides[1], 0)
    else:
        pad_along_height = max(f_height - (channels % strides[1]), 0)
    if timepoints % strides[2] == 0:
        pad_along_width = max(f_width - strides[2], 0)
    else:
        pad_along_width = max(f_width - (timepoints % strides[2]), 0)
    # Finally, the padding on the top, bottom, left and right are:

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return pad_left, pad_right, pad_top, pad_bottom


class EEGNet(nn.Module):
    """
    EEGNet from "EEGNet: A Compact Convolutional Neural Network
    for EEG-based Brain-Computer Interfaces"
    """
    def __init__(self, timepoints: int, channels: int, dropout: float = 0.5, kernel_length: int = 64, f1: int = 8,
                 d: int = 2, f2: int = 16) -> None:
        """
        Initializes an instance of EEGNet.
        Parameters:
            timepoints: the number of time points of data
            channels: the number of channels
            dropout: dropout probability for regularization
            kernel_length: the size of the kernel of the first 2d convolution
            f1: the number of filters to learn for the first convolution
            d: the depth for depthwise and separable convolution
            f2: the number of filters to learn for depthwise and separable
                convolution, must be f1 * d
        """
        assert f2 == d * f1, f"f2 needs to be the product of d and f1." \
                             f"{f2} =/= {d} * {f1}"

        super(EEGNet, self).__init__()
        self.timepoints = timepoints
        self.channels = channels
        self.kernel_length = kernel_length

        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, kernel_length), bias=False),
            nn.BatchNorm2d(f1, False)
        )

        self.block2 = nn.Sequential(
            DepthwiseConv2d(f1, d, (self.channels, 1), bias=False),
            nn.BatchNorm2d(f2, False),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )

        self.block3 = nn.Sequential(
            SeparableConv2d(f2, f2, (1, 16), bias=False),
            nn.BatchNorm2d(f2, False),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )

        self.dense = nn.Linear(f2 * (self.timepoints // 32), 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward the inputs through the model.
        Parameter:
            inputs: the tensor of data
        """
        inputs = inputs.unsqueeze(1)
        input_padding = list(get_padding_sequence(self.channels, self.timepoints, 1, self.kernel_length))
        inputs = F.pad(inputs, input_padding)
        block1_out = self.block1(inputs)
        block2_out = self.block2(block1_out)
        block3_out = self.block3(block2_out)

        return self.dense(block3_out.squeeze(2).view(block3_out.size(0), -1))


class DepthwiseConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 depth_multiplier,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 padding_mode='zeros'
                 ):
        out_channels = in_channels * depth_multiplier
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode
        )


class SeparableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 padding_mode='zeros',
                 depth_multiplier=1,
                 ):
        super().__init__()

        intermediate_channels = in_channels * depth_multiplier
        self.kernel_size = kernel_size
        self.spatial_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=intermediate_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode
        )
        self.point_conv = nn.Conv2d(
            in_channels=intermediate_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, x):
        input_height, input_width = x.size()[2:]
        f_height, f_width = self.kernel_size
        input_padding = get_padding_sequence(input_height, input_width,
                                             f_height, f_width)
        x = F.pad(x, list(input_padding))
        return self.point_conv(self.spatial_conv(x))


class AdaptiveInputNormEEGNet(DN3EEGNet):
    """
    EEGNet with a layer of learned normalization.
    """
    def __init__(self, feat_dim: int, start_gate_iter: int, targets, samples, channels, do=0.25, pooling=8, F1=8, D=2,
                 t_len=65, F2=16, return_features=False) -> None:
        """
        Initializes an EEGNet instance.
        """
        super(AdaptiveInputNormEEGNet, self).__init__(
            targets, samples, channels, do, pooling, F1, D, t_len, F2, return_features)

        self.adaptive_input_norm = AdaptiveInputNorm(feat_dim, start_gate_iter)

    def features_forward(self, x):
        x = self.adaptive_input_norm(x)
        return super().features_forward(x)


class AdaptiveInputNormTIDNet(TIDNet):
    """
    EEGNet with a layer of learned normalization.
    """
    def __init__(self, feat_dim: int, start_gate_iter: int, targets, samples, channels, do=0.4, pooling=20,
                 t_filters=65, return_features=False) -> None:
        """
        Initializes an EEGNet instance.
        """
        super(AdaptiveInputNormTIDNet, self).__init__(
            targets, samples, channels, do=do, pooling=pooling, t_filters=t_filters, return_features=return_features)

        self.adaptive_input_norm = AdaptiveInputNorm(feat_dim, start_gate_iter)

    def features_forward(self, x):
        x = self.adaptive_input_norm(x)
        return super().features_forward(x)


class AdaptiveInputNormEEGNetAuthorDAIN(DN3EEGNet):
    def __init__(self, feat_dim: int, targets, samples, channels, do=0.25, pooling=8, F1=8, D=2,
                 t_len=65, F2=16, return_features=False) -> None:
        """
        Initializes an EEGNet instance.
        """
        super(AdaptiveInputNormEEGNetAuthorDAIN, self).__init__(
            targets, samples, channels, do, pooling, F1, D, t_len, F2, return_features)

        self.adaptive_input_norm = DAIN_Layer(mode="full", input_dim=feat_dim)

    def features_forward(self, x):
        x = self.adaptive_input_norm(x)
        return super().features_forward(x)
