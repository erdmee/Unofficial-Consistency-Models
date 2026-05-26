import torch
from cm.models.unet import UNetModel
from cm.models.precond import ConsistencyPrecond
from cm.training.cd_trainer import CDTrainer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 32
    teacher_ckpt = "checkpoints/edm_cifar10_teacher.pt"

    # =================================================================
    # 1. Build Models 
    # =================================================================
    def create_model():
        unet = UNetModel(in_channels=3, model_channels=128, out_channels=3)
        return ConsistencyPrecond(unet).to(device)

    online_model = create_model()
    target_model = create_model()
    teacher_model = create_model()

    # =================================================================
    # 2. Load Teacher & Initialize 
    # =================================================================
    print(f"[*] Loading Teacher weights from {teacher_ckpt}...")
    teacher_state_dict = torch.load(teacher_ckpt, map_location=device)
    
    # 1) Teacher setting
    teacher_model.load_state_dict(teacher_state_dict)
    teacher_model.requires_grad_(False)
    teacher_model.eval()

    # 2) Student(Online/Target) initialization by Teacher weights (emprical best practice)
    online_model.load_state_dict(teacher_state_dict)
    target_model.load_state_dict(teacher_state_dict)
    target_model.requires_grad_(False)
    target_model.eval()

    # =================================================================
    # 3. Trainer Setup & Start Training
    # =================================================================
    trainer = CDTrainer(
        online_model=online_model,
        target_model=target_model,
        teacher_model=teacher_model,
        data_dir="./data/cifar10/train",
        batch_size=64,
        image_size=image_size,
        lr=1e-4
    )
    
    trainer.train()

if __name__ == "__main__":
    main()