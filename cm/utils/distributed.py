"""
Torchrun-based DDP helpers. GPU-only (CUDA + NCCL).

Launch single-node:
    torchrun --nproc_per_node=8 -m cm.training.train

Single-process runs (no torchrun env vars) still target cuda:0 — every helper
here returns sensible defaults so the same trainer code works for 1 GPU and N
GPUs. There is no CPU fallback by design.
"""

import io
import os
from contextlib import contextmanager
from typing import Iterator, Tuple

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Process group lifecycle
# ---------------------------------------------------------------------------

def is_distributed() -> bool:
    return "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup(backend: str = "nccl") -> Tuple[torch.device, int, int]:
    """
    Initialize the default process group from torchrun env vars.
    GPU-only: requires CUDA.

    Returns:
        (device, local_rank, world_size)
    """
    if is_distributed():
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        local_rank = get_local_rank()
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}"), local_rank, get_world_size()

    torch.cuda.set_device(0)
    return torch.device("cuda:0"), 0, 1


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------

def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce by mean across ranks (no-op when single-process)."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    reduced = tensor.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= get_world_size()
    return reduced


# ---------------------------------------------------------------------------
# Rank-0 read + broadcast helpers
#
# Pattern from openai/consistency_models (cm/dist_util.py): only rank 0
# touches the disk, then the bytes are broadcast to every other rank. This
# matters when (a) the filesystem is slow/networked and (b) we want all
# ranks to start from bit-identical weights regardless of any FS race.
# ---------------------------------------------------------------------------

def sync_params(module: torch.nn.Module, src: int = 0) -> None:
    """
    Broadcast every parameter and buffer from `src` rank to all other ranks.

    Call this AFTER each rank has constructed the model (e.g. loaded weights
    from a teacher checkpoint) and BEFORE wrapping with DDP. Ensures all
    ranks share identical initial state.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    with torch.no_grad():
        for p in module.parameters():
            dist.broadcast(p.data, src=src)
        for b in module.buffers():
            dist.broadcast(b.data, src=src)


def load_state_dict_bcast(path: str, map_location=None, src: int = 0) -> dict:
    """
    Load a checkpoint on `src` rank only, broadcast the raw bytes to all
    other ranks over NCCL, then deserialize everywhere.

    Returns a state_dict equivalent to ``torch.load(path, map_location=...)``
    on every rank. Defaults `map_location` to the calling rank's CUDA device.
    """
    if map_location is None:
        map_location = torch.device(f"cuda:{torch.cuda.current_device()}")

    if not (dist.is_available() and dist.is_initialized()):
        return torch.load(path, map_location=map_location)

    cuda = torch.device("cuda")
    if get_rank() == src:
        with open(path, "rb") as f:
            payload = f.read()
        size = torch.tensor([len(payload)], dtype=torch.long, device=cuda)
    else:
        payload = b""
        size = torch.tensor([0], dtype=torch.long, device=cuda)

    dist.broadcast(size, src=src)

    n = int(size.item())
    buf = torch.empty(n, dtype=torch.uint8, device=cuda)
    if get_rank() == src:
        buf.copy_(torch.frombuffer(bytearray(payload), dtype=torch.uint8))
    dist.broadcast(buf, src=src)

    return torch.load(io.BytesIO(bytes(buf.cpu().numpy())), map_location=map_location)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

@contextmanager
def main_process_first() -> Iterator[None]:
    """
    Force non-main ranks to wait while rank 0 runs the protected block.
    Useful for one-time setup (downloading data, building a cache).
    """
    if get_world_size() == 1:
        yield
        return
    if is_main_process():
        yield
        barrier()
    else:
        barrier()
        yield
