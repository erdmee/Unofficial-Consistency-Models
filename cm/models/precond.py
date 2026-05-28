import torch
import torch.nn as nn

from cm.models.unet import UNetModel


class EDMPrecond(nn.Module):
    """EDM preconditioning (Karras et al. 2022) — no boundary epsilon.

    Use for the teacher network in CD, or to wrap an EDM model directly.
    """
    def __init__(
        self,
        model: UNetModel,
        sigma_data: float = 0.5,
    ):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data

    def forward(self, x, t, y=None):
        t_broadcast = t.view(-1, 1, 1, 1)

        c_in   = 1.0 / (t_broadcast ** 2 + self.sigma_data ** 2).sqrt()
        c_skip = (self.sigma_data ** 2) / (t_broadcast ** 2 + self.sigma_data ** 2)
        c_out  = (t_broadcast * self.sigma_data) / (t_broadcast ** 2 + self.sigma_data ** 2).sqrt()

        # Match openai/consistency_models' rescaled_t (guided-diffusion timestep scale)
        c_noise = 1000 * 0.25 * torch.log(t + 1e-44)

        model_out = self.model(x * c_in, c_noise, y)
        return c_skip * x + c_out * model_out


class ConsistencyPrecond(nn.Module):
    """EDM-style preconditioning that enforces the consistency boundary f(x, ε)=x."""
    def __init__(
        self,
        model: UNetModel,
        sigma_data: float = 0.5,
        epsilon: float = 0.002,
    ):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data
        self.epsilon = epsilon

    def forward(self, x, t, y=None):
        t_broadcast = t.view(-1, 1, 1, 1)

        c_in = 1.0 / (t_broadcast ** 2 + self.sigma_data ** 2).sqrt()

        # c_skip → 1 and c_out → 0 as t → epsilon, enforcing f(x, epsilon) = x.
        c_skip = (self.sigma_data ** 2) / ((t_broadcast - self.epsilon) ** 2 + self.sigma_data ** 2)
        c_out = (self.sigma_data * (t_broadcast - self.epsilon)) / (self.sigma_data ** 2 + t_broadcast ** 2).sqrt()

        c_noise = 1000 * 0.25 * torch.log(t + 1e-44)

        model_out = self.model(x * c_in, c_noise, y)
        return c_skip * x + c_out * model_out
