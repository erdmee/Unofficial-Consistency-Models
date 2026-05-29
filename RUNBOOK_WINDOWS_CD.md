# Runbook — Windows + ImageNet-64 CD 학습

윈도우 PC에서 GPU 인스턴스(Elice 등)에 ImageNet-64 Consistency Distillation 학습을 셋업·실행하는 가이드. 팀원용 self-contained 문서.

전제: 64×64로 사전 리사이즈된 ImageNet 데이터(1000 wnid 폴더)와 teacher EDM 체크포인트(`edm_imagenet64_ema.pt`)를 로컬 윈도우 PC에 이미 보유.

---

## 0. WSL2 강력 권장

WSL2(Windows Subsystem for Linux)를 깔면 rsync/ssh/tmux/scp 모두 Linux처럼 사용 가능. ML 작업엔 거의 필수.

### WSL2 설치 (1회, 관리자 PowerShell)
```powershell
wsl --install
```
재부팅 후 Ubuntu 자동 실행. 사용자/패스워드 설정.

WSL Ubuntu 안에서:
```bash
sudo apt update && sudo apt install -y rsync openssh-client
```

→ 이후 WSL 안에서는 본 레포의 `RUNBOOK.md` (Mac 가이드) 그대로 적용 가능. 본 문서는 **WSL 없이 순수 PowerShell만 사용하는 케이스**를 추가로 다룸.

---

## 1. SSH 셋업 (PowerShell)

### (a) 키 파일 배치 + 권한

Elice 콘솔에서 `.pem` 키 파일 다운로드 후:

```powershell
mkdir $env:USERPROFILE\.ssh -Force
move C:\Users\<유저명>\Downloads\elice-*.pem $env:USERPROFILE\.ssh\
```

권한 잠그기 (Linux의 `chmod 600` 동등):

```powershell
icacls $env:USERPROFILE\.ssh\elice-<your_key>.pem /inheritance:r /grant:r "$($env:USERNAME):(R)"
```

### (b) SSH config alias

`C:\Users\<유저명>\.ssh\config` 파일 만들기 (확장자 없음, 메모장이나 VSCode로):

```
Host elice-cd
    HostName <central-XX.tcp.tunnel.elice.io>
    User elicer
    Port <port>
    IdentityFile C:\Users\<유저명>\.ssh\elice-<your_key>.pem
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

HostName / Port는 Elice 콘솔의 "Use Another Client" 정보 복사.

### (c) 접속 테스트

```powershell
ssh elice-cd
```

처음 접속 시 host key fingerprint 확인 prompt → `yes`.

---

## 2. 인스턴스 1회 셋업

SSH 접속 (`ssh elice-cd`) 후 인스턴스 안에서:

```bash
nvidia-smi
df -h ~
```

GPU 메모리 ≥ 40GB, 디스크 ≥ 200GB 권장.

```bash
export WORKDIR=~/cm
mkdir -p $WORKDIR && cd $WORKDIR
echo "export WORKDIR=$WORKDIR" >> ~/.bashrc
```

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

코드 받기 (GitHub에서 clone):

```bash
git clone https://github.com/erdmee/Unofficial-Consistency-Models.git ~/cm
cd ~/cm
uv sync
source .venv/bin/activate
```

CUDA 작동 확인:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

기대 출력: `2.5.1 True NVIDIA A100 80GB PCIe` 등.

---

## 3. 데이터 업로드 (Windows → 인스턴스)

64×64 리사이즈된 ImageNet 데이터가 `C:\path\to\imagenet64_resized\` 아래에 1000개 wnid 폴더로 있다고 가정.

**기대 구조** (loader가 인식):
```
~/cm/data/imagenet64/train/
├── n01440764/n01440764_10026.png
├── n01440764/n01440764_10027.png
├── n01443537/...
├── ... (총 1000개 wnid 폴더)
└── n15075141/...
```

클래스 라벨은 부모 폴더명(WNID)에서 자동 추출됨.

### 옵션 A — WSL rsync (권장)

WSL 안에서 (Windows 드라이브는 `/mnt/c/`로 마운트):

```bash
rsync -avz --progress /mnt/c/path/to/imagenet64_resized/ elice-cd:cm/data/imagenet64/train/
```

→ 끊겨도 재실행 시 이어 받음. 7GB 기준 인터넷 속도 따라 10–60분.

### 옵션 B — WinSCP GUI (윈도우 친화)

1. WinSCP 설치 (https://winscp.net)
2. 신규 사이트:
   - Protocol: SFTP
   - Host: `<central-XX.tcp.tunnel.elice.io>`
   - Port: `<port>`
   - User: `elicer`
   - Advanced → SSH → Authentication → Private key: 위 `.pem` 파일
     (변환 prompt 뜨면 `.ppk`로 저장 동의)
3. 접속 후 좌측(로컬 Windows) → 우측(원격 `~/cm/data/imagenet64/train/`) 드래그
4. 옵션에서 "Reconnect on disconnect" 켜면 자동 재시도

### 옵션 C — PowerShell scp

```powershell
scp -r C:\path\to\imagenet64_resized\* elice-cd:cm/data/imagenet64/train/
```

단점: 끊기면 처음부터. 자잘한 파일 많을 때 느림. 옵션 A/B가 안정적.

---

## 4. Teacher 체크포인트 업로드 (1.2GB)

`edm_imagenet64_ema.pt` 한 파일.

원격에서 디렉터리 먼저:
```bash
mkdir -p ~/cm/pretrained
```

### WSL rsync
```bash
rsync -avz --progress /mnt/c/path/to/edm_imagenet64_ema.pt elice-cd:cm/pretrained/
```

### WinSCP
GUI에서 `pretrained/` 폴더 만들고 드래그.

### PowerShell scp
```powershell
scp C:\path\to\edm_imagenet64_ema.pt elice-cd:cm/pretrained/
```

---

## 5. 데이터/ckpt 검증 (인스턴스)

```bash
cd ~/cm
ls data/imagenet64/train/ | wc -l
```
→ `1000` 나와야 OK.

```bash
find data/imagenet64/train/ -type f | wc -l
```
→ ~`1281167` (ILSVRC2012 train 전체).

```bash
ls -lh pretrained/edm_imagenet64_ema.pt
```
→ ~`1.2G`.

```bash
python -c "
from PIL import Image
import glob
sample = sorted(glob.glob('data/imagenet64/train/*/*.png'))[0]
img = Image.open(sample)
print(f'{sample}: {img.size} {img.mode}')
"
```
→ `(64, 64) RGB` 나오면 정상.

---

## 6. wandb 키 등록

https://wandb.ai/authorize 에서 발급받은 40자 키:

```bash
echo 'export WANDB_API_KEY=<여기에_40자키>' >> ~/.bashrc
source ~/.bashrc
wandb login --verify
```

기대: `Currently logged in as: <username>`.

wandb 로그인이 안 되면 일단 config에서 `logging.use_wandb: false`로 끄고 진행 가능. 나중에 켜고 resume.

---

## 7. Dry-run (~5–10분)

본 학습 전에 teacher 로딩 + 메모리 + throughput 검증:

```bash
cp configs/imagenet64_cd.yaml configs/imagenet64_cd_dryrun.yaml
sed -i 's/max_steps: 1000000/max_steps: 200/' configs/imagenet64_cd_dryrun.yaml
sed -i 's/ckpt_every: 5000/ckpt_every: 100/' configs/imagenet64_cd_dryrun.yaml
sed -i 's/log_every: 50/log_every: 10/' configs/imagenet64_cd_dryrun.yaml
```

```bash
PYTHONUNBUFFERED=1 python -m cm.training.train --config configs/imagenet64_cd_dryrun.yaml --mode cd
```

확인 포인트:
- `[train] loading teacher state_dict from pretrained/edm_imagenet64_ema.pt` 메시지
- loss / grad_norm / param_norm 숫자로 찍힘 (NaN 없음)
- step 100, 199에 ckpt 저장
- 콘솔에 wandb URL 출력 → 브라우저로 곡선 확인
- 실제 s/step 측정 (A100 PCIe 기준 ~0.4–0.8s 예상)
- `nvidia-smi`로 VRAM 사용량 확인 (25–40GB 예상)

정리:
```bash
ls checkpoints/
rm -rf checkpoints/ wandb/ configs/imagenet64_cd_dryrun.yaml
```

---

## 8. 본 학습 (tmux로 SSH 끊겨도 계속)

```bash
tmux new -s cd
```

(tmux 안에서)
```bash
cd ~/cm && source .venv/bin/activate
echo "WANDB key set? ${WANDB_API_KEY:+yes}"
```

```bash
PYTHONUNBUFFERED=1 python -m cm.training.train --config configs/imagenet64_cd.yaml --mode cd 2>&1 | tee training.log
```

초반 wandb URL 출력되니 브라우저에 북마크.

**tmux detach** (학습은 백그라운드 계속): `Ctrl+B` 누르고 손 떼고 `D`

→ 이제 PowerShell 창 닫아도, Windows 재부팅해도, 학습은 인스턴스 안에서 계속 돌아감. **인스턴스 자체를 Stop/Reimage만 안 하면 됨**.

---

## 9. 모니터링

### wandb 대시보드 (실시간, 권장)

```
https://wandb.ai/<username>/consistency-models/runs/<run_id>
```
loss / grad_norm / param_norm / GPU util 실시간 곡선.

### 원격 SSH 별도 세션

```powershell
ssh elice-cd
```

```bash
tmux attach -t cd          # 학습 화면 다시 보기 (다시 detach: Ctrl+B, D)
tail -f ~/cm/training.log  # 로그 따라가기
watch -n 1 nvidia-smi      # GPU 실시간
ls -la ~/cm/checkpoints/   # ckpt 진행
```

### 로컬에서 한 줄 체크

```powershell
ssh elice-cd 'ls ~/cm/checkpoints/ | tail -3'
```

→ 새 step 파일이 시간 지날수록 늘어나면 살아있음.

---

## 10. 종료 / 재개

### 정상 종료

```powershell
ssh elice-cd
```
```bash
tmux attach -t cd
```
→ `Ctrl+C` 누르면 final ckpt 자동 저장 후 종료.

### 인스턴스 중단/오류 후 재개

```powershell
ssh elice-cd
```
```bash
cd ~/cm && source .venv/bin/activate
LATEST=$(ls -1 checkpoints/step_*.pt | sort | tail -1)
echo "Resuming from $LATEST"
tmux new -s cd
PYTHONUNBUFFERED=1 python -m cm.training.train --config configs/imagenet64_cd.yaml --mode cd --resume $LATEST 2>&1 | tee -a training.log
```

→ wandb 같은 run에 자동으로 이어 로깅됨 (레포에 구현됨).

---

## 11. 최종 산출물 다운로드 (학습 후)

원격 ckpt를 윈도우로:

### WSL rsync
```bash
rsync -avz --progress elice-cd:cm/checkpoints/ /mnt/c/path/to/save/
```

### WinSCP
GUI에서 원격 `~/cm/checkpoints/` → 로컬 폴더로 드래그.

### PowerShell scp
```powershell
scp -r elice-cd:cm/checkpoints C:\path\to\save\
```

샘플 이미지 생성 (원격에서):
```bash
python -m cm.sampling.onestep \
  --ckpt checkpoints/step_XXXXXX.pt \
  --batch_size 64 --image_size 64 \
  --num_classes 1000 --class_id 1 \
  --out_path samples.png
```

---

## 12. 자주 발생하는 문제

| 증상 | 원인 / 해결 |
|---|---|
| `Permissions are too open` SSH 거부 | 키 파일 권한 → `icacls <pem> /inheritance:r /grant:r "$($env:USERNAME):(R)"` |
| `FileNotFoundError: ... edm_imagenet64_ema.pt` | teacher ckpt 미배치 → `pretrained/edm_imagenet64_ema.pt` 확인 |
| 데이터 로드 시 0개 파일 | 경로 잘못 또는 폴더 구조 안 맞음 → `find data/imagenet64/train -type f \| wc -l` 확인 |
| 학습 시작 후 콘솔 로그 안 뜸 | Python 출력 버퍼링 → `PYTHONUNBUFFERED=1` 붙임. wandb 대시보드는 실시간 |
| 첫 step 30–60초 안 끝남 | DataLoader workers + cuDNN warmup. 정상. nvidia-smi로 GPU 사용 확인 |
| OOM (out of memory) | 다른 GPU 프로세스 있나 확인 (`nvidia-smi`). 정 안되면 yaml `batch_size: 64 → 32`로 축소 |
| `wandb: rate limit exceeded` | 15–30분 대기 후 시크릿 모드 + https://wandb.ai/authorize 한 번만 |
| Resume 후 wandb 새 run으로 시작 | ckpt에 `wandb_run_id` 필드 없는 legacy. 신규 학습부터는 자동 이어짐 |
| scp/rsync 도중 끊김 | rsync는 그대로 재실행 (이어 받음). scp는 처음부터 → WinSCP나 rsync 권장 |
| tmux 세션 못 찾음 | `tmux ls`로 목록 확인. 이름 다르면 `tmux attach -t <이름>` |

---

## 부록 — 핵심 파라미터

| 항목 | 값 (`configs/imagenet64_cd.yaml`) |
|---|---|
| 모델 크기 | 296M params (192ch UNet, channel_mult [1,2,3,4]) |
| 해상도 | 64×64 |
| 클래스 조건 | 1000 (class-conditional) |
| Batch size | 64 |
| Learning rate | 8e-6 |
| Mixed precision | fp16 |
| ckpt_every | 5000 |
| max_steps | 1,000,000 |
| Teacher | `pretrained/edm_imagenet64_ema.pt` |
| Sampling EMA decay | 0.999943 |
| Target EMA mu | 0.95 |
| CD num_scales | 40 |

A100 80GB PCIe 단일 GPU 기준 추정 throughput: ~0.4–0.8 s/step.
1M step 소요 시간: ~111–222 시간 (4.6–9.3일). wandb loss plateau 보고 중간 종료 권장.

---

## 한 줄 요약

1) PowerShell에서 SSH config 등록 → 2) WinSCP나 WSL rsync로 `~/cm/data/imagenet64/train/<wnid>/*.png`와 `~/cm/pretrained/edm_imagenet64_ema.pt` 업로드 → 3) 인스턴스에서 `git clone ...repo... ~/cm && cd ~/cm && uv sync && source .venv/bin/activate` → 4) `wandb login` + dry-run으로 검증 → 5) `tmux new -s cd` 후 `PYTHONUNBUFFERED=1 python -m cm.training.train --config configs/imagenet64_cd.yaml --mode cd 2>&1 | tee training.log` → 6) `Ctrl+B, D`로 detach. 끝.
