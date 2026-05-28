# Consistency Models: Dataset/Mode-agnostic 학습 리팩토링

## Context

현재 `cm/training/train.py`(55줄)는 테스트용 placeholder로, CIFAR-10 값들이 하드코딩되어 있고 YAML config를 읽지 않으며 CT 엔트리포인트도 없다. **버린다 — 새로 작성한다.** 한편 데이터 로더(`cm/data/loader.py:22-84`)와 trainer 클래스(`CDTrainer.__init__`, `CTTrainer.__init__`)는 이미 dataset-specific 값을 생성자 인자로 받도록 잘 분리되어 있으므로 그대로 활용한다.

목표: **단일 `train.py` + `--config <path>` + `--mode {cd,ct}`**로 4가지 (dataset × mode) 조합을 코드 변경 없이 갈아끼울 수 있게 만든다. 백본은 우리 ADM 계열 `cm/models/unet.py`로 통일 (NCSN++ 미사용).

| 조합 | class cond | teacher 필요 | 학습 우선순위 |
|---|---|---|---|
| **ImageNet-64 + CD** | conditional (1000) | `pretrained/edm_imagenet64_ema.pt` (확보 완료) | **1순위 (메인)** |
| **CIFAR-10 + CT** | unconditional | 없음 (random init) | **1순위 (메인)** |
| ImageNet-64 + CT | conditional (1000) | 없음 (또는 EDM init 옵션) | **2순위** (자원 여유 시) |
| CIFAR-10 + CD | unconditional | (사용자 별도 준비 필요) | **3순위** (상황 보고 결정) |

**1·2순위는 학습 진행**, **3순위는 일단 보류**. 단 코드와 config는 4 mode 모두 갈아끼울 수 있도록 인프라 완비 (사용자 요구사항).

### Teacher 체크포인트 정책

- **ImageNet-64 EDM teacher**: 우리 UNet과 100% state_dict 매칭 확인 완료 (아래 "검증" 섹션 항목 0).
- **CIFAR-10 EDM teacher**: 현재 미보유. CIFAR-10 CD 학습을 시작하려면 사용자가 `pretrained/edm_cifar10_teacher.pt` 형식의 PyTorch state_dict를 별도로 준비해야 함. config schema는 미리 준비해두지만 실행은 보류.
- **ImageNet-64 CT의 EDM init (옵션)**: `cfg.ct.pretrained_ckpt` 키로 명시 시 같은 `edm_imagenet64_ema.pt`를 inner UNet weights로 복사. CT loss는 teacher forward를 쓰지 않지만 시작 가중치만 EDM에서 가져오는 게 논문 권장.

리팩토링 범위는 최소화 원칙: **신규 파일은 config 3개**(`cifar10_cd.yaml`, `imagenet64_cd.yaml`, `imagenet64_ct.yaml`), 기존 파일 수정은 `train.py`(전면 재작성, ~120줄), `cd_trainer.py`(4줄), `ct_trainer.py`(1줄), `configs/cifar10_ct.yaml`(스키마 마이그레이션)에 한정. UNet/loss/dataloader/precond/utils는 손대지 않는다.

## 변경 사항

### 1. `cm/training/train.py` — **신규 작성** (기존 placeholder 폐기)

argparse로 `--config <path>`, `--mode {cd,ct}`, `--resume <path>` 수신. YAML 로드 후 mode 분기.

파일 내부 헬퍼 (별도 모듈 만들지 않음):
- `load_yaml(path) -> dict`
- `build_unet(cfg, device) -> UNetModel`: 공통 UNet 생성. `num_classes: null` 이면 `None` 전달 → unconditional.
- `build_consistency(cfg, device) -> ConsistencyPrecond`: student/target용. boundary ε 포함.
- `build_edm(cfg, device) -> EDMPrecond`: teacher용. boundary ε **없음** (EDM 표준 preconditioning).

**중요 — Teacher vs Student preconditioning 분리**: inner UNet은 동일하지만 wrapper가 다름.
- Teacher (`EDMPrecond`): `c_skip = σ_d² / (t² + σ_d²)`, boundary 강제 없음.
- Student/Target (`ConsistencyPrecond`): `c_skip = σ_d² / ((t-ε)² + σ_d²)`, `f(x, ε)=x` 강제.

세 모델 모두 같은 EDM teacher state_dict로 초기화 가능 — wrapper 차이는 forward 시 곱해지는 계수일 뿐이고 state_dict 키는 inner UNet에만 존재. 로딩은 `.model.load_state_dict(sd)`로 inner UNet에 직접 주입.

CD 분기:
1. teacher = `build_edm(cfg, device)` — **EDMPrecond로 감싼다**
2. online, target = `build_consistency(cfg, device)` 두 개 — **ConsistencyPrecond**
3. `sd = torch.load(cfg.cd.teacher_ckpt, map_location=device)`; 세 모델 모두 `.model.load_state_dict(sd)`
4. teacher.requires_grad_(False).eval(); target.requires_grad_(False).eval()
5. `CDTrainer(online, target, teacher, ..., num_scales=cfg.cd.num_scales, target_mu=cfg.cd.target_mu, log_every=cfg.logging.log_every).train()`

CT 분기:
1. online, target = `build_consistency(cfg, device)` 두 개 (teacher 없음)
2. (옵션) `cfg.ct.pretrained_ckpt` 명시 시 `.model.load_state_dict(sd)` (EDM init — 논문 권장)
3. `CTTrainer(online, target, ..., s0, s1, mu0, log_every=cfg.logging.log_every).train()`

### 2. `cm/training/cd_trainer.py` — 4줄 수정

논문 Table 3은 CD target EMA μ를 dataset별로 다른 값으로 지정: CIFAR=0(hard copy), ImageNet=0.95. 현재 코드는 `mu=0.999`가 line 116에 하드코딩. logging frequency도 line 119에 `100` 하드코딩.

- `__init__` 시그니처에 `num_scales: int = 18`, `target_mu: float = 0.95`, `log_every: int = 50` 추가 후 저장.
- line 106 `num_scales=18` → `num_scales=self.num_scales`.
- line 116 `update_ema(..., mu=0.999)` → `mu=self.target_mu`.
- line 119 `step % 100 == 0` → `step % self.log_every == 0`.

### 2b. `cm/training/ct_trainer.py` — 1줄 수정

- `__init__` 시그니처에 `log_every: int = 50` 추가 후 저장.
- line 143 `step % 100 == 0` → `step % self.log_every == 0`.

(CT의 `s0/s1/mu0`는 이미 `ct_trainer.py:29-31`에서 파라미터화 완료)

### 3. Config 스키마 — 통합 규칙

공통 키:
- `data: {image_size, channels, data_dir}`
- `model: {model_channels, channel_mult, num_res_blocks, attention_resolutions, num_head_channels OR num_heads, dropout, num_classes}`
  - `num_classes: null` ⇒ unconditional, 정수 ⇒ conditional
  - `attention_resolutions`는 **spatial size** (예: 32×32 입력에서 [16, 8]은 ds=2,4). `unet.py:85`의 `image_size // ds` 비교 의미.
- `training: {loss, batch_size, lr, max_steps}` — `lpips | l2` (학습자원 제약으로 LPIPS가 주력)
- `logging: {ckpt_every, log_every, out_dir}`

Mode-specific:
- CD 전용: `cd.teacher_ckpt` (필수), `cd.num_scales`, `cd.target_mu`
- CT 전용: `ct.s0`, `ct.s1`, `ct.mu0`, `ct.pretrained_ckpt` (옵션)

### 4. 신규 config 3종 — 논문 Table 3 + 공식 launch.sh 기반

논문 Table 3 핵심:

| Hyperparameter | CIFAR-10 CD | CIFAR-10 CT | ImageNet-64 CD | ImageNet-64 CT |
|---|---|---|---|---|
| Learning rate | 4e-4 | 4e-4 | 8e-6 | 8e-6 (논문) / **1e-4 (공식)** |
| Batch size (논문) | 512 | 512 | 2048 | 2048 |
| μ (CD target EMA) | 0 | — | 0.95 | — |
| μ₀ (CT) | — | 0.9 | — | 0.95 |
| s₀ / s₁ | — | 2 / 150 | — | 2 / 200 |
| N (CD) | 18 | — | 40 | — |
| Training iter (논문) | 800k | 800k | 600k | 800k |
| Mixed Precision | No | No | Yes | Yes |
| Dropout | 0.0 | 0.0 | 0.0 | 0.0 |

자원 제약으로 **batch는 64**(단일 A100 80GB), **max_steps는 100k** (검증용)로 축소. Optimizer는 우리 trainer 기본인 AdamW 그대로 사용 (논문 RAdam과 큰 차이 없음).

**`configs/imagenet64_cd.yaml`** — 1순위 메인

```yaml
data: {image_size: 64, channels: 3, data_dir: ./data/imagenet64/train}
model:
  model_channels: 192
  channel_mult: [1, 2, 3, 4]
  num_res_blocks: 3
  attention_resolutions: [32, 16, 8]
  num_head_channels: 64
  dropout: 0.0
  num_classes: 1000
training: {loss: lpips, batch_size: 64, lr: 0.000008, max_steps: 100000}
cd: {teacher_ckpt: pretrained/edm_imagenet64_ema.pt, num_scales: 40, target_mu: 0.95}
logging: {ckpt_every: 5000, log_every: 50, out_dir: runs/imagenet64_cd}
```

**`configs/imagenet64_ct.yaml`** — 2순위. lr은 공식 1e-4 채택 (적은 step 효율). EDM init 옵션은 주석으로.

```yaml
data: {image_size: 64, channels: 3, data_dir: ./data/imagenet64/train}
model:
  model_channels: 192
  channel_mult: [1, 2, 3, 4]
  num_res_blocks: 3
  attention_resolutions: [32, 16, 8]
  num_head_channels: 64
  dropout: 0.0
  num_classes: 1000
training: {loss: lpips, batch_size: 64, lr: 0.0001, max_steps: 100000}
ct:
  s0: 2
  s1: 200
  mu0: 0.95
  pretrained_ckpt: pretrained/edm_imagenet64_ema.pt   # 논문 권장 EDM init (필요 시 주석 처리하면 random init)
logging: {ckpt_every: 5000, log_every: 50, out_dir: runs/imagenet64_ct}
```

**`configs/cifar10_cd.yaml`** — 3순위. `teacher_ckpt` 경로는 placeholder. 사용자가 PyTorch state_dict로 변환한 EDM CIFAR teacher를 그 경로에 두면 그대로 실행 가능.

```yaml
data: {image_size: 32, channels: 3, data_dir: ./data/cifar10/train}
model:
  model_channels: 128
  channel_mult: [1, 2, 2, 2]
  num_res_blocks: 4
  attention_resolutions: [16, 8]
  num_heads: 4
  dropout: 0.0
  num_classes: null
training: {loss: lpips, batch_size: 256, lr: 0.0004, max_steps: 100000}
cd: {teacher_ckpt: pretrained/edm_cifar10_teacher.pt, num_scales: 18, target_mu: 0.0}
logging: {ckpt_every: 5000, log_every: 50, out_dir: runs/cifar10_cd}
```

### 5. `configs/cifar10_ct.yaml` — 스키마 마이그레이션 (1순위)

기존 파일을 위 통합 규칙에 맞춰 정리:

- `model.base_channels` → `model.model_channels`
- `model:` 블록에 `num_classes: null` 추가 (CIFAR-10 CT는 unconditional)
- `schedule.s0`, `schedule.s1` → `ct.s0`, `ct.s1`로 그룹 이동
- `training.total_steps` → `training.max_steps` (값 100000으로)
- `training.ema_mu0: 0.95` → `ct.mu0: 0.9` (**논문 Table 3 값으로 정정**)
- `training.lr: 0.0002` → `training.lr: 0.0004` (**논문 Table 3 값으로 정정**)
- `training.batch_size: 512` → `256` (단일 A100 80GB 메모리 안전 범위)
- `training.pseudo_huber_c` 줄 제거 (pseudo-huber 미지원)
- `logging.log_every: 100` → `50`
- `schedule.sigma_min/sigma_max/sigma_data/rho`는 현재 코드에서 직접 참조되지 않으므로 보존 (별건)
- `attention_resolutions: [16, 8]`은 그대로

## 변경하지 않는 파일

- `cm/models/unet.py` — `num_classes=None` 분기가 이미 구현됨. train.py에서 키워드로 명시만 하면 됨.
- `cm/models/precond.py`, `cm/models/layers.py` (EDMPrecond/ConsistencyPrecond 둘 다 존재 확인됨)
- `cm/data/loader.py`, `cm/data/dataset.py`, `cm/data/transforms.py`
- `cm/training/losses.py`, `cm/training/ema.py`
- `cm/diffusion/*`, `cm/utils/*`, `cm/evaluation/*`, `cm/sampling/*`

## 핵심 파일

- `cm/training/train.py` (**신규 작성** — 기존 55줄 placeholder 폐기)
- `cm/training/cd_trainer.py` (4줄 수정)
- `cm/training/ct_trainer.py` (1줄 수정)
- `configs/cifar10_ct.yaml` (스키마 마이그레이션)
- `configs/cifar10_cd.yaml` (신규, teacher placeholder)
- `configs/imagenet64_cd.yaml` (신규)
- `configs/imagenet64_ct.yaml` (신규)

## 예산 (엘리스 클라우드 — 80만원 한도)

- **인스턴스**: G-NAHP-80 (A100 80GB PCIe 1대), ₩2,000/h + 스토리지 128GiB ₩19.2/h = 실효 **₩2,019/h**
- **1순위 학습** (ImageNet CD 100k + CIFAR CT 100k + 마진): 약 42h → **약 ₩8.5만** (한도의 11%)
- **2순위 추가** (ImageNet CT 100k): +6h → 누적 약 ₩10만
- **3순위 진입** (CIFAR CD): teacher 준비 + 학습 약 10h → 누적 약 ₩12만 (단 teacher 준비 자체가 별건)
- 풀 학습까지 확장해도 합계 약 ₩31만 — 한도 안에서 풍부한 여유
- 학습 안 할 때는 인스턴스 종료, 스토리지만 유지(₩19.2/h ≈ 일 462원)

## 검증

0. **EDM teacher state_dict 매칭** ✅ **(2026-05-28 확인 완료)**
   - 파일: `pretrained/edm_imagenet64_ema.pt` (1.18 GB, 541 keys)
   - 우리 UNet 인스턴스(295.9M params, 위 config 값 그대로)와 `load_state_dict(strict=True)` 통과
   - `missing=0`, `unexpected=0` — 키맵 변환 불필요
   - `EDMPrecond(unet)` / `ConsistencyPrecond(unet)` 둘 다 forward 통과 (64×64×3 출력, class cond 1000 정상)

1. **CIFAR-10 CT smoke**: `label_emb` 미생성 확인, Step 0 로그 `N(k)=2 | μ(k)=0.9 | CT Loss: ...` 출력. (1순위)
2. **ImageNet-64 CD smoke**: teacher state_dict 로드 성공 → `label_emb`(1000) 생성 → Step 0 `CD Loss` 출력. (1순위)
3. **ImageNet-64 CT smoke**: EDM init 적용 → Step 0 `N(k)/μ(k)/CT Loss` 출력. (2순위)
4. **CIFAR-10 CD smoke**: teacher 파일 준비된 경우에만 — `label_emb` 미생성, Step 0 `CD Loss` 출력. (3순위, 보류)
5. **단일 trainer 회귀**: `training.max_steps`를 임시 2로 override해 1·2순위 config가 1 step forward+backward 통과.
6. **기존 테스트 통과**: `pytest tests/` — 모델/sampler/precond/schedule shape 테스트는 변경 없음이므로 모두 green.

## Run Commands

```bash
# === Environment ===
cd "/Users/erdembileg/Desktop/Consistency Models"
source .venv/bin/activate

# === Regression: existing tests must stay green ===
pytest tests/ -q

# === 1순위 학습 (single GPU / CPU) ===

# CIFAR-10 CT (random init, no teacher) — 가장 가벼움, 먼저
python -m cm.training.train --config configs/cifar10_ct.yaml --mode ct

# ImageNet-64 CD (pretrained EDM teacher) — 메인
python -m cm.training.train --config configs/imagenet64_cd.yaml --mode cd

# === 2순위 학습 ===

# ImageNet-64 CT (EDM init from same teacher ckpt, no teacher forward)
python -m cm.training.train --config configs/imagenet64_ct.yaml --mode ct

# === 3순위 (teacher .pt 준비 후에만) ===

# CIFAR-10 CD (requires user-supplied pretrained/edm_cifar10_teacher.pt)
python -m cm.training.train --config configs/cifar10_cd.yaml --mode cd

# === Resume from checkpoint (모든 config에서 동일) ===
python -m cm.training.train \
    --config configs/imagenet64_cd.yaml \
    --mode cd \
    --resume checkpoints/step_005000.pt

# === Multi-GPU distributed (torchrun) — 풀 학습 시 ===
torchrun --standalone --nproc_per_node=4 \
    -m cm.training.train --config configs/imagenet64_cd.yaml --mode cd

torchrun --standalone --nproc_per_node=8 \
    -m cm.training.train --config configs/cifar10_ct.yaml --mode ct
```
