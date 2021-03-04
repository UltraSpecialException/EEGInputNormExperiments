import torch
import torch.nn as nn


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

        feat_std_over_time = (shifted_inputs ** 2).mean(dim=2)
        scaler = self.scale(feat_std_over_time)
        shifted_scaled_inputs = shifted_inputs / scaler.unsqueeze(2)

        if self.curr_iter >= self.start_gate_iter:
            shifted_scaled_summary = shifted_scaled_inputs.mean(dim=2)
            gate = torch.sigmoid(self.gate(shifted_scaled_summary))

            normed_inputs = shifted_scaled_inputs * gate.unsqueeze(2)
        else:
            normed_inputs = shifted_scaled_inputs

        self.curr_iter += 1

        return normed_inputs
