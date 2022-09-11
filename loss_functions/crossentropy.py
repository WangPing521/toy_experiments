import torch
from deepclustering2.loss.kl_losses import _check_reduction_params
from deepclustering2.utils import simplex
from torch import nn, Tensor


class SimplexCrossEntropyLoss(nn.Module):
    def __init__(self, reduction="mean", eps=1e-16) -> None:
        super().__init__()
        _check_reduction_params(reduction)
        self._reduction = reduction
        self._eps = eps

    def forward(self, prob: Tensor, target: Tensor, **kwargs) -> Tensor:
        if not kwargs.get("disable_assert"):
            assert not target.requires_grad
            assert prob.requires_grad
            assert prob.shape == target.shape
            assert simplex(prob)
            assert simplex(target)
        b, c, *_ = target.shape
        ce_loss = (-target * torch.log(prob+self._eps)).sum(1)
        if self._reduction == "mean":
            return ce_loss.mean()
        elif self._reduction == "sum":
            return ce_loss.sum()
        else:
            return ce_loss