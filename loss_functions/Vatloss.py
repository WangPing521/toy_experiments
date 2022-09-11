import contextlib
from typing import Tuple, Union, List, Optional, Dict, OrderedDict

import torch
import torch.nn as nn
from deepclustering2.utils import simplex, assert_list
from torch import Tensor


class KL_div(nn.Module):
    r"""
    KL(p,q)= -\sum p(x) * log(q(x)/p(x))
    where p, q are distributions
    p is usually the fixed one like one hot coding
    p is the target and q is the distribution to get approached.

    reduction (string, optional): Specifies the reduction to apply to the output:
    ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
    ``'mean'``: the sum of the output will be divided by the number of
    elements in the output, ``'sum'``: the output will be summed.
    """

    def __init__(self, reduction="mean", eps=1e-16, weight: Union[List[float], Tensor] = None, verbose=True):
        super().__init__()
        self._eps = eps
        self._reduction = reduction
        self._weight: Optional[Tensor] = weight
        if weight is not None:
            assert isinstance(weight, (list, Tensor)), type(weight)
            if isinstance(weight, list):
                assert assert_list(lambda x: isinstance(x, (int, float)), weight)
                self._weight = torch.Tensor(weight).float()
            else:
                self._weight = weight.float()
            # normalize weight:
            self._weight = self._weight / self._weight.sum()
        if verbose:
            print(f"Initialized {self.__class__.__name__} \nwith weight={self._weight} and reduction={self._reduction}.")

    def forward(self, prob: Tensor, target: Tensor, **kwargs) -> Tensor:
        if not kwargs.get("disable_assert"):
            assert prob.shape == target.shape
            assert simplex(prob), prob
            assert simplex(target), target
            assert not target.requires_grad
            assert prob.requires_grad
        b, c, *hwd = target.shape
        kl = (-target * torch.log((prob + self._eps) / (target + self._eps)))
        if self._weight is not None:
            assert len(self._weight) == c
            weight = self._weight.expand(b, *hwd, -1).transpose(-1, 1).detach()
            kl *= weight.to(kl.device)
        kl = kl.sum(1)
        if self._reduction == "mean":
            return kl.mean()
        elif self._reduction == "sum":
            return kl.sum()
        else:
            return kl


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, "track_running_stats"):
            m.track_running_stats ^= True

    # let the track_running_stats to be inverse
    model.apply(switch_attr)
    # return the model
    yield
    # let the track_running_stats to be inverse
    model.apply(switch_attr)


def _l2_normalize(d: torch.Tensor) -> torch.Tensor:
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True)  # + 1e-8
    ones_ = torch.ones(d.shape[0], device=d.device)
    assert torch.allclose(d.view(d.shape[0], -1).norm(dim=1), ones_, rtol=1e-3)
    return d


class VATLoss(nn.Module):
    def __init__(
        self, xi=10.0, eps=1.0, prop_eps=0.25, ip=1, distance_func=KL_div()
    ):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 2)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip
        self.prop_eps = prop_eps
        self.distance_func = distance_func

    def forward(self, model, x: torch.Tensor):
        """
        We support the output of the model would be a simplex.
        :param model:
        :param x:
        :return:
        """
        with torch.no_grad():
            # pred = model(x).softmax(1)
            pred = torch.softmax(model(x) / 10, dim=1)
        assert simplex(pred)

        # prepare random unit tensor
        d = torch.randn_like(x, device=x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                # pred_hat = model(x + self.xi * d).softmax(1)
                pred_hat = torch.softmax(model(x + self.xi * d) / 10, dim=1)
                adv_distance = self.distance_func(pred_hat, pred)
                adv_distance.backward()
                d = _l2_normalize(d.grad)

            # calc LDS
            if isinstance(self.eps, torch.Tensor):
                # a dictionary is given
                bn, *shape = x.shape
                basic_view_shape: Tuple[int, ...] = (bn, *([1] * len(shape)))
                r_adv = d * self.eps.view(basic_view_shape).expand_as(d) * self.prop_eps
            elif isinstance(self.eps, (float, int)):
                r_adv = d * self.eps * self.prop_eps
            else:
                raise NotImplementedError(
                    f"eps should be tensor or float, given {self.eps}."
                )

            # pred_hat = model(x + r_adv).softmax(1)
            pred_hat = torch.softmax(model(x + r_adv) / 10, dim=1)
            lds = self.distance_func(pred_hat, pred)

        return lds


