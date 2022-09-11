import torch


def _l2_normalize(d: torch.Tensor) -> torch.Tensor:
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True)  # + 1e-8
    ones_ = torch.ones(d.shape[0], device=d.device)
    assert torch.allclose(d.view(d.shape[0], -1).norm(dim=1), ones_, rtol=1e-3)
    return d


d = torch.randn(2, 3, 3)
d = torch.where(d<0, torch.Tensor([2]), torch.Tensor([1]))
print(d)
d_norm = _l2_normalize(d)
print(d_norm)

