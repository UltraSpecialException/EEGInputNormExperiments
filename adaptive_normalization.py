import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AdaptiveInputNorm(nn.Module):
    """
    Implementation of Deep Adaptive Input Normalization for Time-Series Forecasting.
    https://arxiv.org/pdf/1902.07892.pdf
    """

    def __init__(self, feat_dim: int, start_gate_iter: int) -> None:
        """
        Initializes an instance of the DAIN module.
        """
        super(AdaptiveInputNorm, self).__init__()
        self.shift = nn.Linear(feat_dim, feat_dim, bias=False)
        self.scale = nn.Linear(feat_dim, feat_dim, bias=False)
        self.gate = nn.Linear(feat_dim, feat_dim, bias=True)
        self.curr_iter = 0
        self.start_gate_iter = start_gate_iter

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Run the inputs through the adaptive normalization module.:
        """
        # inputs shape (batch, features, time)
        feat_avg_over_time = inputs.mean(dim=2)    # batch x features
        shifter = self.shift(feat_avg_over_time)
        shifted_inputs = inputs - shifter.unsqueeze(2)

        return shifted_inputs

        # feat_std_over_time = (shifted_inputs ** 2).mean(dim=2)
        # scaler = self.scale(feat_std_over_time)
        # shifted_scaled_inputs = shifted_inputs / scaler.unsqueeze(2)
        #
        # if self.curr_iter >= self.start_gate_iter:
        #     shifted_scaled_summary = shifted_scaled_inputs.mean(dim=2)
        #     gate = torch.sigmoid(self.gate(shifted_scaled_summary))
        #
        #     normed_inputs = shifted_scaled_inputs * gate.unsqueeze(2)
        # else:
        #     normed_inputs = shifted_scaled_inputs
        #
        # self.curr_iter += 1
        #
        # return normed_inputs


class DAIN_Layer(nn.Module):
    def __init__(self, mode='adaptive_avg', mean_lr=0.00001, gate_lr=0.001, scale_lr=0.00001, input_dim=144):
        super(DAIN_Layer, self).__init__()
        print("Mode = ", mode)

        self.mode = mode
        self.mean_lr = mean_lr
        self.gate_lr = gate_lr
        self.scale_lr = scale_lr

        # Parameters for adaptive average
        self.mean_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.mean_layer.weight.data = torch.FloatTensor(data=np.eye(input_dim, input_dim))

        # Parameters for adaptive std
        self.scaling_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.scaling_layer.weight.data = torch.FloatTensor(data=np.eye(input_dim, input_dim))

        # Parameters for adaptive scaling
        self.gating_layer = nn.Linear(input_dim, input_dim)

        self.eps = 1e-8

    def forward(self, x):
        # Expecting  (n_samples, dim,  n_feature_vectors)

        # Nothing to normalize
        if self.mode == None:
            pass

        # Do simple average normalization
        elif self.mode == 'avg':
            avg = torch.mean(x, 2)
            avg = avg.resize(avg.size(0), avg.size(1), 1)
            x = x - avg

        # Perform only the first step (adaptive averaging)
        elif self.mode == 'adaptive_avg':
            avg = torch.mean(x, 2)
            adaptive_avg = self.mean_layer(avg)
            adaptive_avg = adaptive_avg.resize(adaptive_avg.size(0), adaptive_avg.size(1), 1)
            x = x - adaptive_avg

        # Perform the first + second step (adaptive averaging + adaptive scaling )
        elif self.mode == 'adaptive_scale':

            # Step 1:
            avg = torch.mean(x, 2)
            adaptive_avg = self.mean_layer(avg)
            adaptive_avg = adaptive_avg.resize(adaptive_avg.size(0), adaptive_avg.size(1), 1)
            x = x - adaptive_avg

            # Step 2:
            std = torch.mean(x ** 2, 2)
            std = torch.sqrt(std + self.eps)
            adaptive_std = self.scaling_layer(std)
            adaptive_std[adaptive_std <= self.eps] = 1

            adaptive_std = adaptive_std.resize(adaptive_std.size(0), adaptive_std.size(1), 1)
            x = x / (adaptive_std)

        elif self.mode == 'full':

            # Step 1:
            avg = torch.mean(x, 2)
            adaptive_avg = self.mean_layer(avg)
            adaptive_avg = adaptive_avg.resize(adaptive_avg.size(0), adaptive_avg.size(1), 1)
            x = x - adaptive_avg

            # # Step 2:
            std = torch.mean(x ** 2, 2)
            std = torch.sqrt(std + self.eps)
            adaptive_std = self.scaling_layer(std)
            adaptive_std[adaptive_std <= self.eps] = 1

            adaptive_std = adaptive_std.resize(adaptive_std.size(0), adaptive_std.size(1), 1)
            x = x / adaptive_std

            # Step 3:
            avg = torch.mean(x, 2)
            gate = F.sigmoid(self.gating_layer(avg))
            gate = gate.resize(gate.size(0), gate.size(1), 1)
            x = x * gate

        else:
            assert False

        return x
