import torch
import torch.nn as nn


class PINN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, depth=4):
        super().__init__()

        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 1)]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
