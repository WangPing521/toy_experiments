import contextlib
from typing import Tuple, Union, List, Optional, Dict, OrderedDict

import torch
import torch.nn as nn
from deepclustering2.utils import simplex, assert_list
from torch import Tensor
from torch.distributions import Bernoulli
from loss_functions.constraint_loss import local_cons_binary_convex, symetry_reward
from tool.independent_functions import average_list

device = 'cuda'
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


class Local_cons(nn.Module):
    def __init__(self, num_sample=3):
        super().__init__()
        self._num = num_sample

    def forward(self, prob: Tensor, cons_name='symetry', **kwargs) -> Tensor:
        sample_container = Bernoulli(prob[:, 1, :, :])
        sample_list, sample_list_trans = [], []
        prob_list, prob_list_trans = [], []
        # sample distribution
        for n in range(self._num):
            sample_n = sample_container.sample()
            sample_list.append(sample_n)

            map_bg = torch.where(sample_n == 0, torch.Tensor([1]).to(device), torch.Tensor([0]).to(device))
            pred = map_bg * prob[:, 0, :, :] + sample_n * prob[:, 1, :, :]
            pred = 1 - pred
            prob_list.append(pred)
        # for reducing the computational complexity
        assert len(sample_list) == len(prob_list)
        for img in range(prob.shape[0]):
            for i in range(len(prob_list)-1):
                if i == 0:
                    sample_imgs = torch.cat([sample_list[i][img].unsqueeze(0).unsqueeze(0), sample_list[i+1][img].unsqueeze(0).unsqueeze(0)], dim=1)
                    prob_imgs = torch.cat([prob_list[i][img].unsqueeze(0).unsqueeze(0), prob_list[i+1][img].unsqueeze(0).unsqueeze(0)], dim=1)
                else:
                    sample_imgs = torch.cat([sample_imgs, sample_list[i+1][img].unsqueeze(0).unsqueeze(0)], dim=1)
                    prob_imgs = torch.cat([prob_imgs, prob_list[i+1][img].unsqueeze(0).unsqueeze(0)], dim=1)
            sample_list_trans.append(sample_imgs.transpose(1, 0))
            prob_list_trans.append(prob_imgs.transpose(1, 0))

        if cons_name in ['symetry']:
            all_shapes_list, symmetry_errors_list = symetry_reward(sample_list_trans)
            reward_list = [torch.where(error==torch.Tensor([0]), torch.tensor([-1]).to(device), error) for error in symmetry_errors_list]
        else:
            reward_list = local_cons_binary_convex(sample_list_trans, scale=3)

        assert len(reward_list) == len(prob_list_trans)
        cons_localloss = []
        for sample_idx in range(len(reward_list)):
            cons_localloss.append(
                (-reward_list[sample_idx].unsqueeze(1) * torch.log(prob_list_trans[sample_idx] + 0.000001)).mean())

        cons_loss = average_list(cons_localloss)

        return all_shapes_list, symmetry_errors_list, cons_loss


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


class consVATLoss(nn.Module):
    def __init__(
        self, xi=10.0, eps=1.0, prop_eps=0.25, ip=1, consweight=0.5, mode='vat', distance_func=KL_div(), constraint_func=Local_cons()
    ):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 2)
        """
        super(consVATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip
        self.prop_eps = prop_eps
        self.distance_func = distance_func
        self.local_constriant = constraint_func
        self.mode = mode
        self.consweight = consweight

    def forward(self, model, x: torch.Tensor):
        """
        We support the output of the model would be a simplex.
        :param model:
        :param x:
        :return:
        """
        with torch.no_grad():
            # pred = model(x).softmax(1)
            pred = torch.softmax(model(x), dim=1)
        assert simplex(pred)

        # prepare random unit tensor
        d = torch.randn_like(x, device=x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = torch.softmax(model(x + self.xi * d), dim=1)
                adv_distance = self.distance_func(pred_hat, pred)
                all_shapes_list, shape_error_list, adv_cons = self.local_constriant(pred_hat)
                adv_distance = adv_distance + self.consweight * adv_cons
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

            if self.mode in ['cons']:
                lds = 0
                pred_hat = torch.softmax(model(x), dim=1)
                all_shapes_list, shape_error_list, cons = self.local_constriant(pred_hat)
            elif self.mode in ['vat']:
                cons = 0
                pred_hat = torch.softmax(model(x + r_adv), dim=1)
                lds = self.distance_func(pred_hat, pred)
            elif self.mode in ['cat']:
                pred_hat = torch.softmax(model(x + r_adv), dim=1)
                lds = self.distance_func(pred_hat, pred)
                all_shapes_list, shape_error_list, cons = self.local_constriant(pred_hat)

        return all_shapes_list, shape_error_list, lds, cons


