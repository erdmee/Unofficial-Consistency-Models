# Unofficial Consistency Models

PyTorch implementation of *Consistency Models* (Song et al., 2023) for CIFAR-10
and class-conditional ImageNet 64×64. Both consistency distillation (CD) from
an EDM teacher and teacher-free consistency training (CT) are supported.

## Setup

Python 3.11 and `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
```

`pyproject.toml` pins `torch==2.5.1` and `torchvision==0.20.1`. On Linux the
wheels are pulled from `https://download.pytorch.org/whl/cu121`, which works
with any NVIDIA driver that supports CUDA ≥ 12.1. macOS gets the default PyPI
build (CPU/MPS) — useful for running tests, not for training.

## Data

CIFAR-10:

```bash
python -m cm.data.download --dataset cifar10 --data_dir ./data
```

This unpacks the PyTorch CIFAR-10 download into
`./data/cifar10/{train,val}/<class>/<idx>.png`.

ImageNet 64×64 follows the EDM convention. Download ILSVRC2012 from
[Kaggle](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data),
then center-crop to square and Lanczos-resize to 64×64. The loader accepts
either `<root>/<class>/<img>.png` or flat files prefixed with the WNID
(`n01440764_2708.JPEG`); the class label is inferred from the parent directory
or filename prefix.

For CD, place an EDM teacher checkpoint at the path given by `cd.teacher_ckpt`
in the config (e.g. `pretrained/edm_imagenet64_ema.pt`). The file should be a
state dict matching `cm.models.unet.UNetModel`.

## Training

```bash
python -m cm.training.train --config configs/cifar10_ct.yaml      --mode ct
python -m cm.training.train --config configs/cifar10_cd.yaml      --mode cd
python -m cm.training.train --config configs/imagenet64_ct.yaml   --mode ct
python -m cm.training.train --config configs/imagenet64_cd.yaml   --mode cd
```

`--resume path/to/step_XXXXXX.pt` continues from a saved checkpoint.

CT optionally initializes from `ct.pretrained_ckpt` (e.g. an EDM-pretrained
UNet). Whether or not a pretrained file is loaded, the target and sampling-EMA
models are synchronized to the online model before training starts.

A separate **sampling EMA** is maintained alongside the **target EMA** (paper
§4). `cm.sampling.onestep` loads the sampling EMA by default.

### Defaults

| | CIFAR-10 CD | CIFAR-10 CT | ImageNet64 CD | ImageNet64 CT |
|---|---|---|---|---|
| Model channels | 128 | 128 | 192 | 192 |
| Channel mult | 1,2,2,2 | 1,2,2,2 | 1,2,3,4 | 1,2,3,4 |
| Res blocks per stage | 4 | 4 | 3 | 3 |
| Attention resolutions | 16, 8 | 16, 8 | 32, 16, 8 | 32, 16, 8 |
| Class-conditional | no | no | yes (1000) | yes (1000) |
| Optimizer | RAdam, wd=0 | RAdam, wd=0 | RAdam, wd=0 | RAdam, wd=0 |
| Learning rate | 4e-4 | 4e-4 | 8e-6 | 1e-4 |
| Batch size | 256 | 256 | 64 | 64 |
| Mixed precision | fp32 | fp32 | fp16 | fp16 |
| Target EMA μ | 0.0 | schedule | 0.95 | schedule |
| Sampling EMA | 0.9999 | 0.9999 | 0.999943 | 0.999943 |
| N (CD) / s₀→s₁ (CT) | 18 | 2 → 150 | 40 | 2 → 200 |
| Max steps | 100k | 100k | 100k | 100k |

Batch sizes and step counts are reduced from the paper (paper uses 512/2048
and 600–800k steps). Edit the YAML to scale up if you have the compute budget.

### Logging

`wandb` is wired into the trainer. Run `wandb login` once, then loss, lr, and
(for CT) `N(k)` and `μ(k)` are streamed to the project named in
`logging.wandb_project`. Set `logging.use_wandb: false` to disable.

## Sampling

```bash
python -m cm.sampling.onestep \
  --ckpt checkpoints/step_050000.pt \
  --batch_size 64 --image_size 32 \
  --out_path samples.png
```

For a class-conditional ImageNet64 model, pass `--num_classes 1000 --class_id <id>`.
Without `--class_id`, every sample comes from class 0.

The multi-step sampler in `cm.sampling.multistep` takes a descending list of
intermediate noise levels τ. Paper-recommended schedules for CIFAR-10:

- NFE = 2 → `[0.821]`
- NFE = 4 → `[24.4, 5.84, 0.9]`

## Project layout

```
cm/
  models/      UNet + EDM/Consistency preconditioning
  diffusion/   Karras sigmas, EMA schedules, Heun solver
  training/    CD/CT trainers, loss functions, entrypoint
  sampling/    one-step and multi-step samplers
  evaluation/  FID via TF-Inception reference weights
  data/        dataset, loader, transforms, CIFAR-10 downloader
configs/       per-dataset × per-mode YAML
tests/         shape and numerical sanity tests
```

## References

- Song, Y., Dhariwal, P., Chen, M., Sutskever, I. *Consistency Models*. ICML 2023.
  [arXiv:2303.01469](https://arxiv.org/abs/2303.01469).
  [Official code](https://github.com/openai/consistency_models).
- Karras, T., Aittala, M., Aila, T., Laine, S. *Elucidating the Design Space of
  Diffusion-Based Generative Models*. NeurIPS 2022.
  [arXiv:2206.00364](https://arxiv.org/abs/2206.00364).
  [Official code](https://github.com/NVlabs/edm).
- Dhariwal, P., Nichol, A. *Diffusion Models Beat GANs on Image Synthesis*.
  NeurIPS 2021.
  [arXiv:2105.05233](https://arxiv.org/abs/2105.05233).
