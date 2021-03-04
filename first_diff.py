from dn3.transforms.instance import InstanceTransform
import torch


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
    
