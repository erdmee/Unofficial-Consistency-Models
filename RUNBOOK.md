# Runbook — Consistency Models 학습 실행

CIFAR-10 CT (또는 CD) 기준 end-to-end 실행 가이드. ImageNet-64는 데이터 준비 단계만 추가.

대부분의 명령은 원격 GPU 인스턴스 안에서 동일. **로컬 OS(Mac vs Windows)에 따라 다른 부분만 분기**.

---

## 0. 사전 준비 (1회)

| 항목 | 비고 |
|---|---|
| GPU 인스턴스 (Elice 등) | 본 가이드는 Elice `G-NAHP-80` (1× A100 80GB PCIe) 기준 |
| SSH 키 파일 (.pem) | 인스턴스 콘솔에서 다운로드 |
| Wandb 계정 + API key | https://wandb.ai/authorize 에서 발급 |
| 로컬 코드베이스 | 이 레포 |

ImageNet-64를 쓸 경우:
- Kaggle 계정 + `kaggle.json` API token
- Kaggle competition rules 페이지에서 Accept (CLI 불가)
- Teacher EDM 체크포인트 → `pretrained/edm_imagenet64_ema.pt`

---

## 1. 로컬 셸 + SSH 설정

### Mac

```bash
# ~/.ssh/config 에 alias 추가 (인스턴스 콘솔의 Quick Connect 정보 참고)
cat >> ~/.ssh/config <<'EOF'

Host elice2
	HostName <central-XX.tcp.tunnel.elice.io>
	User elicer
	Port <port>
	IdentityFile ~/.ssh/elice/<your_key.pem>
	ServerAliveInterval 60
	ServerAliveCountMax 3
EOF
```

```bash
chmod 600 ~/.ssh/elice/<your_key.pem>
```

확인:
```bash
ssh elice2
```

### Windows

**옵션 A (권장): WSL2 사용**. WSL2 안에서는 Mac과 동일하게 진행. `~/.ssh/config` 동일하게 작성.

**옵션 B: PowerShell + OpenSSH 내장 클라이언트**. Win10 1809+ 기본 포함.

```powershell
# %USERPROFILE%\.ssh\config 작성 (메모장 등)
Host elice2
    HostName <central-XX.tcp.tunnel.elice.io>
    User elicer
    Port <port>
    IdentityFile C:\Users\<you>\.ssh\elice\<your_key.pem>
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

키 파일 권한:
```powershell
icacls C:\Users\<you>\.ssh\elice\<your_key.pem> /inheritance:r /grant:r "$($env:USERNAME):(R)"
```

확인:
```powershell
ssh elice2
```

### Ghostty 터미널 사용자 공통 (Mac/Linux)

원격에서 `tput: unknown terminal "xterm-ghostty"` 등 경고 나오면 1회만:

```bash
infocmp -x | ssh elice2 -- tic -x -
```

---

## 2. 원격 인스턴스 초기 셋업 (1회)

SSH 접속 후 (`ssh elice2`):

```bash
nvidia-smi
```

```bash
df -h ~
```

```bash
export WORKDIR=~/cm
mkdir -p $WORKDIR && cd $WORKDIR
echo "export WORKDIR=$WORKDIR" >> ~/.bashrc
```

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version
```

---

## 3. 코드 업로드 (로컬 → 원격)

### Mac (rsync)

**Mac 새 터미널**에서:
```bash
cd "/Users/<you>/Desktop/Consistency Models" && rsync -avz --progress --exclude '.venv' --exclude '__pycache__' --exclude '.git' --exclude '/data' --exclude '/checkpoints' --exclude '/wandb' --exclude '/pretrained' --exclude '/runs' --exclude '/outputs' --exclude '*.pyc' --exclude '.DS_Store' --exclude '.pytest_cache' ./ elice2:cm/
```

> ImageNet-64 CD/CT 돌릴 때는 `--exclude '/pretrained'` 제거 (teacher ckpt도 같이 보내야 함). 또는 별도로 `scp pretrained/edm_imagenet64_ema.pt elice2:cm/pretrained/`.

### Windows

**WSL2 사용 시**: 위 Mac 명령과 동일 (WSL2 안에서 rsync 사용).

**PowerShell만 사용 시**: rsync가 없으므로 옵션 두 가지.

옵션 1) **scp** (단일 폴더 전체):
```powershell
cd "C:\path\to\Consistency Models"
scp -r -i C:\Users\<you>\.ssh\elice\<your_key.pem> -P <port> -o "StrictHostKeyChecking=no" `
  cm configs tests pyproject.toml uv.lock README.md .python-version `
  elicer@<central-XX.tcp.tunnel.elice.io>:cm/
```

옵션 2) **WinSCP / FileZilla GUI**: Quick Connect 정보로 접속해서 위 파일/폴더 드래그.

옵션 3) (가장 깔끔) **git push / git clone**: 로컬 변경을 GitHub 등에 push → 원격에서 `git clone`.

---

## 4. 원격 의존성 설치 + 데이터 (1회)

원격 SSH 창에서:

```bash
cd ~/cm
```

```bash
uv sync
```

```bash
source .venv/bin/activate
```

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

기대: `2.5.1 True NVIDIA A100 80GB PCIe`

### CIFAR-10 데이터

```bash
python -m cm.data.download --dataset cifar10 --data_dir ./data
```

### (선택) ImageNet-64 데이터

원본을 로컬에서 미리 64×64 PNG로 변환한 후 업로드 권장 (인스턴스 시간 절약):

```bash
# 로컬에서
python -m cm.data.download --dataset imagenet_raw --raw_dir ./raw_imagenet
python -m cm.data.download --dataset imagenet64 --source_dir ./raw_imagenet/ILSVRC/Data/CLS-LOC/train --data_dir ./data --num_workers 8
```

그 다음 `./data/imagenet64/` 디렉터리를 위의 rsync에 포함시켜 업로드 (excludes 중 `/data` 제거).

---

## 5. wandb 키 등록

원격에서:

```bash
echo 'export WANDB_API_KEY=<여기에_40자키>' >> ~/.bashrc
source ~/.bashrc
wandb login --verify
```

기대: `Currently logged in as: <username>`

---

## 6. Dry-run (1회, ~3분)

본 학습 전 NaN/throughput/ckpt 동작 검증:

```bash
cp configs/cifar10_ct.yaml configs/cifar10_ct_dryrun.yaml
sed -i 's/max_steps: 1000000/max_steps: 200/' configs/cifar10_ct_dryrun.yaml
sed -i 's/ckpt_every: 5000/ckpt_every: 100/' configs/cifar10_ct_dryrun.yaml
sed -i 's/log_every: 50/log_every: 10/' configs/cifar10_ct_dryrun.yaml
```

```bash
PYTHONUNBUFFERED=1 python -m cm.training.train --config configs/cifar10_ct_dryrun.yaml --mode ct
```

확인:
- loss 숫자로 찍히고 NaN 없음
- step 100, 199에 ckpt 저장
- wandb URL 콘솔에 출력 → 브라우저로 곡선 확인
- 실제 s/step 측정 (시간 / 200)

정리:
```bash
rm -rf checkpoints/ wandb/ configs/cifar10_ct_dryrun.yaml
```

---

## 7. 본 학습 (tmux로 SSH 끊겨도 계속)

원격에서:

```bash
tmux new -s ct
```

(tmux 안에서)
```bash
cd ~/cm && source .venv/bin/activate
echo "WANDB key set? ${WANDB_API_KEY:+yes}"
```

CIFAR-10 CT 실행:
```bash
PYTHONUNBUFFERED=1 python -m cm.training.train --config configs/cifar10_ct.yaml --mode ct 2>&1 | tee training.log
```

> ImageNet-64 CD는 `--config configs/imagenet64_cd.yaml --mode cd`. `pretrained/edm_imagenet64_ema.pt` 사전 배치 필수.

**tmux detach**: 키보드 `Ctrl+B` → 손 떼고 `D`

이제 Mac/Windows 터미널 닫아도 학습 계속됨.

---

## 8. 모니터링

### Wandb 대시보드 (실시간 권장)

브라우저로 학습 시작 시 콘솔에 출력된 URL:
```
https://wandb.ai/<username>/consistency-models/runs/<run_id>
```
→ loss / grad_norm / param_norm / GPU util 실시간 곡선.

### 원격 SSH 별도 세션에서

```bash
ssh elice2
```

```bash
# 학습 화면 다시 보기
tmux attach -t ct
# 빠지기: Ctrl+B → D
```

```bash
# 로그 따라가기
tail -f ~/cm/training.log
```

```bash
# GPU 실시간
watch -n 1 nvidia-smi
```

```bash
# ckpt 진행 상황
ls -la ~/cm/checkpoints/
```

### 로컬에서 한 줄 체크

```bash
ssh elice2 'ls ~/cm/checkpoints/ | tail -3'
```

→ 새 step 파일이 시간 지날수록 늘어나면 살아있음.

---

## 9. 종료 / 재개

### 학습 정상 종료

```bash
ssh elice2
tmux attach -t ct
```
```
Ctrl+C
```

→ final ckpt 자동 저장 후 종료.

### 인스턴스 중단/오류 후 재개

```bash
ssh elice2
cd ~/cm && source .venv/bin/activate
LATEST=$(ls -1 checkpoints/step_*.pt | sort | tail -1)
echo "Resuming from $LATEST"
tmux new -s ct
PYTHONUNBUFFERED=1 python -m cm.training.train --config configs/cifar10_ct.yaml --mode ct --resume $LATEST 2>&1 | tee -a training.log
```

→ wandb run 자동으로 이어짐 (같은 곡선에 추가).

---

## 10. 최종 산출물 다운로드

학습 종료 후 로컬로 ckpt 가져오기:

### Mac / WSL2 / Linux

```bash
rsync -avz --progress elice2:cm/checkpoints/ ./final_checkpoints/
```

### Windows PowerShell (rsync 없음)

```powershell
scp -r -i C:\Users\<you>\.ssh\elice\<your_key.pem> -P <port> `
  elicer@<central-XX.tcp.tunnel.elice.io>:cm/checkpoints `
  C:\path\to\final_checkpoints
```

이미지 샘플 생성 (로컬 또는 원격, 둘 다 가능):

```bash
python -m cm.sampling.onestep \
  --ckpt final_checkpoints/step_XXXXXX.pt \
  --batch_size 64 --image_size 32 \
  --out_path samples.png
```

---

## 11. 자주 발생하는 문제

| 증상 | 원인 / 해결 |
|---|---|
| `wandb: rate limit exceeded` 로그인 안 됨 | 짧은 시간 내 다중 시도 → 15–30분 대기 후 시크릿 모드 + https://wandb.ai/authorize 한 번만 |
| 학습 시작 후 콘솔에 로그 안 뜸 | Python 출력 버퍼링. `PYTHONUNBUFFERED=1` 붙이거나 wandb 대시보드로 확인 |
| `ModuleNotFoundError: No module named 'cm.data'` | rsync에서 `--exclude 'data'`가 `cm/data/`도 제외함. `--exclude '/data'`로 anchor |
| 첫 step 30–60초 안 끝남 | DataLoader workers + cuDNN warmup. 정상. `nvidia-smi`로 GPU 사용 확인 |
| ckpt 디스크 가득참 | `keep_last_steps`(default 100_000) + `keep_milestone_every`(default 100_000) 자동 정리됨. 그래도 부족하면 `cm/training/{ct,cd}_trainer.py`에서 값 수정 |
| Resume 후 wandb 새 run으로 시작됨 | ckpt에 `wandb_run_id` 필드 없는 legacy ckpt. 신규 학습부터는 자동 이어짐 |
| `tput: unknown terminal "xterm-ghostty"` | Ghostty terminfo가 원격에 없음. `infocmp -x \| ssh elice2 -- tic -x -` 1회 |
| 코드 변경 후 원격 반영 | Mac에서 동일 rsync 다시 실행 (incremental) |

---

## 부록: 핵심 파라미터 빠른 참조

| | CIFAR-10 CT | CIFAR-10 CD | ImageNet-64 CT | ImageNet-64 CD |
|---|---|---|---|---|
| Model channels | 128 | 128 | 192 | 192 |
| Params | ~70M | ~70M | ~296M | ~296M |
| Batch size (config) | 256 | 256 | 64 | 64 |
| LR | 4e-4 | 4e-4 | 1e-4 | 8e-6 |
| Mixed precision | fp32 | fp32 | fp16 | fp16 |
| Teacher 필요 | ❌ | ✅ EDM ckpt | (선택) EDM init | ✅ EDM ckpt |
| ckpt_every | 5000 | 5000 | 5000 | 5000 |
| max_steps (config) | 1,000,000 | 1,000,000 | 1,000,000 | 1,000,000 |

A100 80GB PCIe 단일 GPU 기준 추정 throughput:
- CIFAR-10 CT/CD (fp32): ~0.15–0.25 s/step
- ImageNet-64 CT/CD (fp16): ~0.4–0.8 s/step
