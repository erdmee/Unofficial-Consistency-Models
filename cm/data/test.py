import os
import torch
from torchvision.utils import save_image

# 앞서 우리가 작성한 모듈들을 불러옵니다.
from download import prepare_cifar10
from loader import create_data_loader

def main():
    data_dir = "./data"
    
    # ==========================================
    # Step 1. 데이터 다운로드 및 준비 (downloader.py)
    # ==========================================
    print("=== Step 1: Prepare CIFAR-10 Dataset ===")
    prepare_cifar10(data_dir)
    
    # 다운로드가 완료되면 이미지들이 저장된 경로를 지정합니다.
    train_dir = os.path.join(data_dir, "cifar10", "train")
    
    # ==========================================
    # Step 2. DataLoader 생성 (loader.py)
    # ==========================================
    print("\n=== Step 2: Create DataLoader ===")
    data_generator = create_data_loader(
        data_dir=train_dir,
        batch_size=16,          # 테스트 확인용이므로 16장만 뽑아봅니다.
        image_size=32,          # CIFAR-10 원본 해상도
        class_cond=True,        # 폴더명을 읽어 클래스 라벨(y) 생성
        deterministic=False,    # Shuffle 활성화
        random_crop=False,      # CIFAR-10은 작아서 CenterCrop으로 진행
        random_flip=True,       # 좌우 반전 활성화
        num_workers=2
    )
    
    # ==========================================
    # Step 3. 데이터 로더 검증 (1 Batch Fetch)
    # ==========================================
    print("\n=== Step 3: Fetch a Batch & Verify Tensors ===")
    # 무한 제너레이터이므로 next()를 사용해 첫 배치를 가져옵니다.
    images, out_dict = next(data_generator)
    labels = out_dict["y"]
    
    print(f"[*] Image Tensor Shape: {images.shape}")  # 예상: [16, 3, 32, 32]
    print(f"[*] Label Tensor Shape: {labels.shape}")  # 예상: [16]
    print(f"[*] Label Values:       {labels.tolist()}")
    
    # Consistency Models은 이미지가 [-1, 1] 범위로 정규화되어 있어야 합니다.
    print(f"[*] Data Range (Min/Max): {images.min().item():.4f} to {images.max().item():.4f}")
    
    # ==========================================
    # Step 4. 시각적 검증 (이미지로 저장)
    # ==========================================
    print("\n=== Step 4: Save Visual Verification Image ===")
    # 현재 이미지는 [-1.0, 1.0] 범위이므로, torchvision.utils.save_image를 위해 
    # [0.0, 1.0] 범위로 복원(De-normalization)해줍니다.
    images_denorm = (images + 1.0) / 2.0
    
    save_path = "test_batch.png"
    save_image(images_denorm, save_path, nrow=4)
    print(f"[*] Success! Saved a grid of 16 images to '{save_path}'.")
    print("[*] 프로젝트 폴더에서 이미지를 열어 Random Flip 등이 잘 적용되었는지 확인해보세요!")

if __name__ == "__main__":
    main()