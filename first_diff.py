from dn3.transforms.instance import InstanceTransform
import torch
from typing import Optional
import pandas


class FirstDifference(InstanceTransform):
    """
    Returns the first differences of a data array.
    """
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the first-difference of the values of the tensor <x> along
        the last axis.
        """
        return x[..., 1:] - x[..., :-1]

    def new_sequence_length(self, old_sequence_length):
        return old_sequence_length - 1


class BrownianMotion(FirstDifference):
    """
    Transform the data by normalizing it as if it is generated by a Brownian Motion random process.
    """
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the first difference from <x> and then zscore normalize it.
        """
        first_diff = super().__call__(x)
        mean = first_diff.mean()
        std = first_diff.std()

        brownian_normed = (first_diff - mean) / std
        return brownian_normed


class StepEWMZScore(FirstDifference):
    """
    A Transformation for EEG data.

    The following steps are taken:
        1. First Difference
        2. For entry at time t, collect the exponentially weighted moving average and standard deviation of the window
           t - w
        3. Perform z-score normalization for each entry at time t using the EWMA and EWMSD collected in step 2.
    """

    def __init__(self, window: int, timesteps: int) -> None:
        """
        Initializes an instance of the transform.
        """
        super(StepEWMZScore, self).__init__()
        self.alpha = 2 / (float(window) + 1)

        self.t = timesteps

        forget_weights = torch.ones(self.t, self.t) * (1 - self.alpha)
        forget_strength = torch.ones(self.t, self.t) * torch.arange(0, -self.t, -1) + torch.arange(self.t)[:, None]

        self.weights = torch.tril(forget_weights ** forget_strength, 0)
        self.weights_sum = self.weights.sum(dim=1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the procedure as described in the class' docstring.
        """
        x = super().__call__(x)
        n, c, _ = x.size()[1:]
        x_flat = x.view(-1, self.t)

        ewma = (x_flat @ self.weights.unsqueeze(-1)).squeeze(-1).transpose(0, 1)
        ewma /= self.weights_sum
        ewma = ewma.view(n, c, self.t)

        bias = self.weights_sum ** 2 / (self.weights_sum ** 2 - (self.weights ** 2).sum(dim=1))

        squared_diff_ewma = (x.unsqueeze(2) - ewma.unsqueeze(-1)) ** 2
        weighted_squared_diff_ewma = (self.weights * squared_diff_ewma).sum(dim=3) / self.weights_sum

        ewmstd = (bias * weighted_squared_diff_ewma) ** 0.5
        ewmstd[ewmstd.isnan()] = 1

        return (x - ewma) / ewmstd
