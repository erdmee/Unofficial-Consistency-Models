import torch
import torch.nn as nn

@torch.no_grad()
def heun_solver(teacher_model: nn.Module, x_t: torch.Tensor, t_current: torch.Tensor, t_next: torch.Tensor):
    """
    Performs a single step of the Heun ODE solver using the pre-trained teacher model.
    Moves the noisy image from t_current to t_next.
    """
    # 1. First Euler step
    denoised_1 = teacher_model(x_t, t_current)
    d_1 = (x_t - denoised_1) / t_current.view(-1, 1, 1, 1)
    x_tmp = x_t + d_1 * (t_next - t_current).view(-1, 1, 1, 1)
    
    # 2. Second Euler step (Correction)
    denoised_2 = teacher_model(x_tmp, t_next)
    d_2 = (x_tmp - denoised_2) / t_next.view(-1, 1, 1, 1)
    
    # 3. Final Heun combination
    x_t_next = x_t + (d_1 + d_2) * (t_next - t_current).view(-1, 1, 1, 1) / 2.0
    return x_t_next