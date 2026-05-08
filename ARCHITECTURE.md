# Architecture

Unofficial PyTorch implementation of *Consistency Models* (Song et al. 2023, [arXiv:2303.01469](https://arxiv.org/abs/2303.01469)).

This document is the single source of truth for module boundaries, ownership, and the interfaces that two contributors must agree on before writing code.

---

## 1. Module overview

```
cm/
├── models/        # Network backbone + EDM preconditioning (defines f_θ)
├── diffusion/     # Karras σ-schedule, N(k), Heun solver (CD teacher step)
├── training/      # Losses, EMA target network, CT/CD training loops
├── sampling/      # 1-step and multistep generation from a trained f_θ
├── data/          # Datasets and image transforms
├── evaluation/    # FID / Inception
└── utils/         # Distributed, logging, checkpointing (leaf — depended on by all)
```

`scripts/` holds entrypoints (`train_ct.py`, `train_cd.py`, `train_edm.py`, `sample.py`).
`configs/` holds YAML run configs.
`tests/` holds interface-level tests (shape, boundary condition, EMA decay).

---

## 2. Ownership (two-person split)

| Track | Person A — *Training / diffusion math* | Person B — *Models / data / sampling* |
|---|---|---|
| Owns | `cm/diffusion/`, `cm/training/`, `scripts/train_*.py` | `cm/models/`, `cm/data/`, `cm/sampling/`, `cm/evaluation/`, `scripts/sample.py` |
| Core responsibilities | σ-schedule, N(k), μ(k) EMA decay, loss `d(·,·)`, CT/CD loops | U-Net `F_θ`, EDM preconditioning `f_θ`, datasets, generation, FID |
| External deps | LPIPS package | torchvision, scipy (FID), Inception weights |

Shared (decide together, freeze early):
- `cm/utils/` (logging, distributed, checkpoint)
- `configs/` schema
- The four interfaces in §4 below

---

## 3. Dependency graph

```
                      ┌────────────────────────┐
                      │ configs/*.yaml         │
                      └──────────┬─────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │   scripts/train_ct.py , train_cd.py , sample.py             │
  └────┬──────────────┬───────────────┬──────────────┬──────────┘
       │              │               │              │
       ▼              ▼               ▼              ▼
  ┌────────┐    ┌────────────┐  ┌────────────┐  ┌─────────────┐
  │ data/  │    │ training/  │  │ sampling/  │  │ evaluation/ │
  │        │    │ ct_trainer │  │ onestep    │  │ fid         │
  │        │    │ cd_trainer │  │ multistep  │  │ inception   │
  └───┬────┘    └─────┬──────┘  └──────┬─────┘  └──────┬──────┘
      │               │                │                │
      │     ┌─────────┼────────────────┼────────────────┘
      │     │         │                │
      │     ▼         ▼                ▼
      │  ┌────────┐ ┌──────────────────────┐
      │  │training│ │  models/precond.py   │  ← f_θ(x,σ): EDM precond + boundary
      │  │losses, │ │  (the consistency fn)│
      │  │ema     │ └──────────┬───────────┘
      │  └────┬───┘            │
      │       │                ▼
      │       │           ┌──────────┐
      │       │           │ models/  │
      │       │           │ unet.py  │  ← F_θ raw backbone
      │       │           └────┬─────┘
      │       │                │
      │       │                ▼
      │       │           ┌──────────┐
      │       │           │ models/  │
      │       │           │ layers.py│
      │       │           └──────────┘
      │       │
      │       ▼
      │  ┌──────────────────┐
      │  │ diffusion/       │  ← σ_min, σ_max, ρ, t_i, N(k)
      │  │ karras_schedule  │  ← Heun 1-step (CD teacher only)
      │  │ solvers          │
      │  └──────────────────┘
      │
      ▼
  ┌──────────────────┐
  │ utils/           │  ← imported everywhere; never imports anything in cm/
  │ distributed,     │
  │ logging,         │
  │ checkpoint       │
  └──────────────────┘
```

### Hard rules
1. `cm/models/` and `cm/diffusion/` **must not import each other**. They are pure modules.
2. `cm/training/` is the only place that combines `models + diffusion + data`.
3. `cm/sampling/` depends only on `cm/models/precond` (a trained `f_θ`) and `cm/diffusion/karras_schedule`.
4. `cm/utils/` is a leaf — it does not import from any other `cm/` subpackage.
5. `configs/` is consumed by `scripts/` only; library modules accept plain Python args, not config objects.

---

## 4. Frozen interfaces (agree first, code after)

These four signatures are the contract between A and B. Once these are merged, both people can work with mocks and integrate later.

### 4.1 `cm/models/unet.py` — owned by B
```python
class UNet(nn.Module):
    def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
        """Raw backbone F_θ. x: (B,C,H,W), sigma: (B,). Returns (B,C,H,W)."""
```

### 4.2 `cm/models/precond.py` — owned by B
```python
class ConsistencyModel(nn.Module):
    """f_θ(x, σ) = c_skip(σ)·x + c_out(σ)·F_θ(c_in(σ)·x, c_noise(σ))

    Boundary condition: f_θ(x, σ_min) == x  (verified in tests/test_precond_boundary.py)
    """
    def __init__(self, backbone: UNet, sigma_data: float, sigma_min: float): ...
    def forward(self, x: Tensor, sigma: Tensor) -> Tensor: ...
```

### 4.3 `cm/diffusion/karras_schedule.py` — owned by A
```python
def karras_sigmas(N: int, sigma_min: float, sigma_max: float, rho: float) -> Tensor:
    """Returns the N+1 boundary points t_0=σ_min .. t_N=σ_max (Karras et al. 2022, Eq. 5)."""

def n_schedule(k: int, total_steps: int, s0: int, s1: int) -> int:
    """N(k) — number of discretization steps at training iter k. CT-specific."""

def mu_schedule(k: int, total_steps: int, mu0: float, s0: int, s1: int) -> float:
    """μ(k) — EMA decay for the target network. CT-specific."""
```

### 4.4 `cm/data/*.py` — owned by B
```python
class CIFAR10(torch.utils.data.Dataset):
    def __getitem__(self, i: int) -> dict:
        # {"image": Tensor[C,H,W] in [-1, 1]}
```

---

## 5. Training data flow

### Consistency Training (CT, no teacher)
```
x_0 ~ data
n   ~ Uniform{1, …, N(k)-1}
z   ~ N(0, I)
t_n, t_{n+1} = karras_sigmas(N(k))[n], [n+1]
x_high = x_0 + t_{n+1} · z
x_low  = x_0 + t_n     · z         # same z — key CT trick
pred_online = f_θ      (x_high, t_{n+1})
pred_target = f_θ⁻ (x_low,  t_n)   # stop-grad, EMA copy
loss        = d(pred_online, pred_target)
optim.step();  ema.update(target ← online, μ(k))
```

### Consistency Distillation (CD, with EDM teacher)
Same as CT, but `x_low` is replaced by one Heun step from `x_high` using the frozen teacher:
```
x_low = HeunStep(teacher, x_high, t_{n+1} → t_n)   # in cm/diffusion/solvers
```
Everything else (loss, EMA, optimizer) is identical to CT — meaning `cd_trainer.py` should mostly reuse `ct_trainer.py` logic.

### Sampling
- **1-step**: `x_T ~ N(0, σ_max² I);  x_0 ≈ f_θ(x_T, σ_max)`
- **Multistep**: alternate `x_n = f_θ(x, t_n)` and re-noise to a smaller `t_{n-1}`, for a fixed schedule of timesteps `τ_1 > τ_2 > … > τ_M`.

---

## 6. Suggested order of work

1. **Together (1 sitting):** finalize §4 interfaces and `cm/utils/checkpoint.py` schema.
2. **Parallel:**
   - A: `karras_schedule` → `losses` → `ema` → `ct_trainer` (mock UNet)
   - B: `layers` → `unet` → `precond` (mock dataset) → `data/cifar10` → `sampling`
3. **Together:** wire `scripts/train_ct.py`, run on CIFAR-10, verify FID trend.
4. **Parallel:**
   - A: `solvers` (Heun) + `cd_trainer` + `train_edm.py`
   - B: `evaluation/fid` + `evaluation/inception` + `sample.py` polish
5. **Together:** scale to LSUN, hyperparameter tuning.

---

## 7. Out-of-scope for v1
- Multi-node distributed training (single-node DDP is fine)
- Latent-space variant (LCM)
- Mixed precision beyond `torch.amp.autocast` defaults
