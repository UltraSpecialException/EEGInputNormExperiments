from dn3.transforms.instance import InstanceTransform
import torch


class FixedScale(InstanceTransform):
    """
    Perform Min-Max normalization to scale the data into the [0, 1] range.
    """
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mins = x.min(dim=-1)[0]
        maxs = x.max(dim=-1)[0]

        scaled = (x - mins.unsqueeze(-1)) / (maxs - mins).unsqueeze(-1)
        scaled[torch.isnan(scaled)] = 0

        return scaled


class ZScore(InstanceTransform):
    """
    Perform ZScore normalization by subtracting the mean and dividing by the standard deviation.
    """
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1)
        std = x.std(dim=-1)
        std[std == 0] = 1

        normed = (x - mean.unsqueeze(-1)) / std.unsqueeze(-1)

        return normed
