import torch
import torch.nn as nn

from cm.models.unet import UNetModel

class ConsistencyPrecond(nn.Module):
    """
    EDM-style preconditioning wrapper for Consistency Models.
    Enforces the boundary condition f(x, epsilon) = x by design.
    """
    def __init__(
        self,
        model: UNetModel,
        sigma_data: float = 0.5, # Assume image data is normalized and has std of ~0.5
        epsilon: float = 0.002,  # t_min (boundary time)
    ):
        """
        Args:
            model (UNetModel): The underlying 2D U-Net architecture.
            sigma_data (float): Standard deviation of the data distribution.
            epsilon (float): The minimum noise level (boundary).
        """
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data
        self.epsilon = epsilon

    def forward(self, x, t, y = None):
        """
        Args:
            x (torch.Tensor): Input noisy image of shape (B, C, H, W)
            t (torch.Tensor): 1-D timestep/noise level tensor of shape (B,)
            y (torch.Tensor, optional): Class labels of shape (B,)
            
        Returns:
            torch.Tensor: Denoised image of shape (B, C, H, W)
        """
        # 1. Broadcast timestep tensor to match image dimensions (B, 1, 1, 1)
        t_broadcast = t.view(-1, 1, 1, 1)
        
        # 2. Calculate preconditioning coefficients
        # c_in: Scales the input so that the network receives unit variance data
        c_in = 1.0 / (t_broadcast ** 2 + self.sigma_data ** 2).sqrt()
        
        # c_skip: Enforces boundary condition. Approaches 1 as t -> epsilon
        c_skip = (self.sigma_data ** 2) / ((t_broadcast - self.epsilon) ** 2 + self.sigma_data ** 2)
        
        # c_out: Enforces boundary condition. Approaches 0 as t -> epsilon
        c_out = (self.sigma_data * (t_broadcast - self.epsilon)) / (self.sigma_data ** 2 + t_broadcast ** 2).sqrt()
        
        # 3. Calculate timestep conditioning (EDM log-scale)
        # We pass this modified noise level to the U-Net's time embedding
        c_noise = 1000 * 0.25 * torch.log(t + 1e-44) # Add tiny epsilon to prevent log(0) just in case
        
        # 4. Pass scaled inputs to the underlying U-Net
        scaled_x = x * c_in
        model_out = self.model(scaled_x, c_noise, y)
        
        # 5. Combine using the skip connection to enforce the boundary
        out = c_skip * x + c_out * model_out
        
        return out