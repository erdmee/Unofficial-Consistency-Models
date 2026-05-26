import os
import torch
from torchvision.utils import save_image
import argparse

# Import our custom modules
from cm.models.unet import UNetModel
from cm.models.precond import ConsistencyPrecond
from cm.utils.checkpoint import load_checkpoint

@torch.no_grad()
def generate_one_step(
    model: torch.nn.Module, 
    batch_size: int, 
    image_size: int, 
    device: torch.device, 
    sigma_max: float = 80.0
):
    """
    Performs 1-step generation using a trained Consistency Model.
    
    Args:
        model (nn.Module): The trained target model (EMA model).
        batch_size (int): Number of images to generate.
        image_size (int): Resolution of the images (e.g., 32 for CIFAR-10).
        device (torch.device): Target device (cpu or cuda).
        sigma_max (float): The maximum noise level (T) used during training.
                           For Karras schedules, this is typically 80.0.
                           
    Returns:
        torch.Tensor: Generated images normalized to [-1, 1].
    """
    model.eval()
    
    # 1. Sample pure noise from N(0, sigma_max^2)
    # The initial state x_T is scaled by the maximum noise level
    shape = (batch_size, 3, image_size, image_size)
    x_T = torch.randn(*shape, device=device) * sigma_max
    
    # 2. Create a timestep tensor filled with sigma_max (T)
    t_tensor = torch.full((batch_size,), sigma_max, device=device)
    
    # 3. 1-Step Forward Pass
    # The Consistency Model directly maps (x_T, T) to the clean image x_0
    x_0 = model(x_T, t_tensor)
    
    # 4. Clamp the output to valid image range [-1.0, 1.0]
    x_0 = torch.clamp(x_0, -1.0, 1.0)
    
    return x_0

def main():
    parser = argparse.ArgumentParser(description="1-Step Sampling for Consistency Models")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the trained checkpoint (.pt)")
    parser.add_argument("--batch_size", type=int, default=64, help="Number of images to generate")
    parser.add_argument("--image_size", type=int, default=32, help="Image resolution")
    parser.add_argument("--out_path", type=str, default="sample_1step.png", help="Output image path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Generating on device: {device}")

    # ==============================================================================
    # 1. Initialize the Model Architecture
    # ==============================================================================
    unet = UNetModel(
        in_channels=3,
        model_channels=128,
        out_channels=3,
    )
    # Consistency Models always generate using the EMA (Target) model for best quality
    target_model = ConsistencyPrecond(unet).to(device)

    # ==============================================================================
    # 2. Load the Trained Weights
    # ==============================================================================
    print(f"[*] Loading checkpoint from {args.ckpt}...")
    # We only need the 'ema_model' for inference. We pass None for online model and optimizer.
    step, _ = load_checkpoint(
        load_path=args.ckpt,
        ema_model=target_model,
        model=None,
        optimizer=None,
        device=str(device)
    )
    print(f"[*] Successfully loaded EMA model from step {step}.")

    # ==============================================================================
    # 3. Generate Images (1-Step)
    # ==============================================================================
    print("[*] Performing 1-step generation...")
    # Generate images in [-1, 1] range
    generated_images = generate_one_step(
        model=target_model,
        batch_size=args.batch_size,
        image_size=args.image_size,
        device=device,
        sigma_max=80.0
    )

    # ==============================================================================
    # 4. Save the Result
    # ==============================================================================
    # De-normalize from [-1, 1] to [0, 1] for saving
    images_denorm = (generated_images + 1.0) / 2.0
    
    save_image(images_denorm, args.out_path, nrow=8)
    print(f"[*] Success! Saved {args.batch_size} images to '{args.out_path}'.")

if __name__ == "__main__":
    main()