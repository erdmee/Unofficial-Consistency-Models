# Unofficial Consistency Models

![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
[![arXiv](https://img.shields.io/badge/arXiv-2303.01469-b31b1b)](https://arxiv.org/abs/2303.01469)

A from-scratch PyTorch reimplementation of *Consistency Models* (Song et al., 2023): generative
models that produce an image in a single network evaluation, with optional few-step refinement. Both
training routes are supported: consistency distillation (CD) from an EDM teacher, and teacher-free
consistency training (CT). Targets are CIFAR-10 and class-conditional ImageNet 64x64.

## Overview

A diffusion model turns noise into data by integrating a probability-flow ODE, which costs many
network evaluations. A consistency model learns a function that maps any point on a trajectory
directly back to its clean origin, so generation collapses to a single evaluation. The boundary
condition (the function is the identity at the smallest noise level) is built into the preconditioning
rather than learned; the teacher uses plain EDM preconditioning. Noise levels follow the Karras
schedule.

Two ways to train the same function:

- **Consistency Distillation (CD).** Two adjacent points on a trajectory are connected by one Heun
  step of a frozen EDM teacher. The online model sees the high-noise point and is pulled toward the
  target model's output at the teacher-denoised point.
- **Consistency Training (CT).** No teacher. The two points are built from the same noise sample,
  which is an unbiased one-sample estimate of the same step. The discretization count is annealed over
  training.

The target that supplies the regression signal is an EMA of the online weights. A separate sampling
EMA is kept for generation, and the samplers load it by default. One-step sampling evaluates the model
once; the multi-step sampler alternately denoises and re-noises over a short list of intermediate
noise levels.

## Setup

Python 3.11 and `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
```

`pyproject.toml` pins `torch==2.5.1` and `torchvision==0.20.1`. On Linux the wheels come from the
CUDA 12.1 index; macOS gets the default CPU/MPS build, which is fine for tests but not for training.

## Data

CIFAR-10:

```bash
python -m cm.data.download --dataset cifar10 --data_dir ./data
```

This unpacks CIFAR-10 into `./data/cifar10/{train,val}/<class>/<idx>.png`.

ImageNet 64x64 follows the EDM convention: download ILSVRC2012, center-crop to square, and resize to
64x64. The loader accepts either `<root>/<class>/<img>.png` or flat files prefixed with the WNID; the
class label is taken from the parent directory or the filename prefix.

For CD, place an EDM teacher checkpoint at the path given by `cd.teacher_ckpt` in the config (for
example `pretrained/edm_imagenet64_ema.pt`). The file is a state dict matching
`cm.models.unet.UNetModel`.

## Training

```bash
python -m cm.training.train --config configs/cifar10_ct.yaml    --mode ct
python -m cm.training.train --config configs/cifar10_cd.yaml    --mode cd
python -m cm.training.train --config configs/imagenet64_ct.yaml --mode ct
python -m cm.training.train --config configs/imagenet64_cd.yaml --mode cd
```

`--resume path/to/step_XXXXXX.pt` continues from a checkpoint. CT can optionally initialize from
`ct.pretrained_ckpt`. In all cases the target and sampling-EMA models are synchronized to the online
model before training starts. The samplers load the sampling EMA by default.

### Defaults

| | CIFAR-10 CT | CIFAR-10 CD | ImageNet64 CT | ImageNet64 CD |
|---|---|---|---|---|
| Model channels | 128 | 128 | 192 | 192 |
| Channel mult | 1,2,2,2 | 1,2,2,2 | 1,2,3,4 | 1,2,3,4 |
| Res blocks per stage | 4 | 4 | 3 | 3 |
| Attention resolutions | 16, 8 | 16, 8 | 32, 16, 8 | 32, 16, 8 |
| Class-conditional | no | no | yes (1000) | yes (1000) |
| Optimizer | RAdam, wd=0 | RAdam, wd=0 | RAdam, wd=0 | RAdam, wd=0 |
| Learning rate | 4e-4 | 4e-4 | 1e-4 | 8e-6 |
| Batch size | 256 | 256 | 64 | 64 |
| Precision | fp32 | fp32 | fp16 | fp16 |
| Loss | LPIPS | LPIPS | LPIPS | LPIPS |

Batch sizes and step counts are reduced from the paper (which uses batch 512 to 2048 over 600k to
800k steps). Edit the YAML to scale up with more compute.

### Logging

`wandb` is wired into the trainer. Run `wandb login` once; loss, learning rate, and (for CT) the
schedules are streamed to the project in `logging.wandb_project`. Set `logging.use_wandb: false` to
disable.

## Sampling

```bash
python -m cm.sampling.onestep \
  --ckpt checkpoints/step_050000.pt \
  --batch_size 64 --image_size 32 \
  --out_path samples.png
```

For a class-conditional ImageNet64 model, pass `--num_classes 1000 --class_id <id>`. Without
`--class_id`, samples come from class 0. The multi-step sampler in `cm.sampling.multistep` takes a
descending list of intermediate noise levels; paper schedules for CIFAR-10 are `[0.821]` for NFE 2 and
`[24.4, 5.84, 0.9]` for NFE 4.

## Project layout

```
cm/
  models/      UNet + EDM/Consistency preconditioning
  diffusion/   Karras sigmas, EMA/discretization schedules, Heun solver
  training/    CD/CT trainers, loss functions, entrypoint
  sampling/    one-step and multi-step samplers
  evaluation/  FID via the reference Inception weights
  data/        dataset, loader, transforms, CIFAR-10 downloader
configs/       per-dataset, per-mode YAML
tests/         shape and numerical sanity tests
```

## References

- Song, Y., Dhariwal, P., Chen, M., Sutskever, I. *Consistency Models*. ICML 2023.
  [arXiv:2303.01469](https://arxiv.org/abs/2303.01469).
  [Official code](https://github.com/openai/consistency_models).
- Karras, T., Aittala, M., Aila, T., Laine, S. *Elucidating the Design Space of Diffusion-Based
  Generative Models*. NeurIPS 2022. [arXiv:2206.00364](https://arxiv.org/abs/2206.00364).
  [Official code](https://github.com/NVlabs/edm).
- Dhariwal, P., Nichol, A. *Diffusion Models Beat GANs on Image Synthesis*. NeurIPS 2021.
  [arXiv:2105.05233](https://arxiv.org/abs/2105.05233).

## Citation

```bibtex
@inproceedings{song2023consistency,
  title     = {Consistency Models},
  author    = {Song, Yang and Dhariwal, Prafulla and Chen, Mark and Sutskever, Ilya},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2023}
}
```
