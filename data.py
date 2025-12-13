import torch


def sample_domain(n_samples):
    S = torch.rand(n_samples, 1)
    v = torch.rand(n_samples, 1)
    t = torch.rand(n_samples, 1)

    return torch.cat([S, v, t], dim=1)
