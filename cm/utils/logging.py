"""
Training logger.

Pattern adopted from openai/guided-diffusion's `logger.py`:
    - logkv(k, v)       — register the latest value of k (overwrites).
    - logkv_mean(k, v)  — accumulate; output is the mean since last dump.
    - dumpkvs(step)     — flush every registered key to stdout / log.txt /
                          progress.csv, then reset.

The trainer calls `logkv_mean` every step for noisy metrics (loss, grad norm),
`logkv` for slow-changing scalars (lr), and `dumpkvs` every N steps. This
gives you one CSV row per dump with the mean over the interval, which is
what you want for a clean training curve.

Implementation here is class-based (not module-global like guided-diffusion)
so multiple loggers / multiple runs can coexist in one process.
"""

import csv
import os
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple


class _RunningMean:
    __slots__ = ("sum", "count")

    def __init__(self) -> None:
        self.sum: float = 0.0
        self.count: int = 0

    def add(self, v: float) -> None:
        self.sum += float(v)
        self.count += 1

    def value(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0


class Logger:
    """
    Usage:
        logger = Logger(log_dir="runs/cd_cifar10", is_main_process=trainer.is_main_process)

        for step in range(max_steps):
            ...
            logger.logkv_mean("loss", loss.item())
            logger.logkv_mean("grad_norm", gn)
            logger.logkv("lr", opt.param_groups[0]["lr"])
            if step % 100 == 0:
                logger.dumpkvs(step)
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        is_main_process: bool = True,
        csv_name: str = "progress.csv",
        text_name: str = "log.txt",
    ):
        self.log_dir = log_dir
        self.is_main_process = is_main_process

        # latest scalar per key (logkv)
        self._latest: "OrderedDict[str, float]" = OrderedDict()
        # running mean per key (logkv_mean)
        self._means: "OrderedDict[str, _RunningMean]" = OrderedDict()

        self._t0 = time.time()
        self._last_dump_t: Optional[float] = None
        self._last_dump_step: Optional[int] = None

        self._csv_path: Optional[str] = None
        self._txt_path: Optional[str] = None
        self._csv_fieldnames: Optional[list] = None
        if self.is_main_process and self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            self._csv_path = os.path.join(self.log_dir, csv_name)
            self._txt_path = os.path.join(self.log_dir, text_name)

    # ----- registration -------------------------------------------------

    def logkv(self, key: str, value: Any) -> None:
        if not self.is_main_process:
            return
        self._latest[key] = float(value)

    def logkv_mean(self, key: str, value: Any) -> None:
        if not self.is_main_process:
            return
        if key not in self._means:
            self._means[key] = _RunningMean()
        self._means[key].add(float(value))

    # ----- flush --------------------------------------------------------

    def dumpkvs(self, step: int) -> Dict[str, float]:
        """Print + persist all registered kvs at `step`, then reset means."""
        if not self.is_main_process:
            return {}

        row: "OrderedDict[str, float]" = OrderedDict()
        row["step"] = float(step)
        for k, rm in self._means.items():
            row[k] = rm.value()
        for k, v in self._latest.items():
            row[k] = v

        it_per_sec = self._step_rate(step)
        if it_per_sec is not None:
            row["it/s"] = it_per_sec
        row["elapsed"] = time.time() - self._t0

        self._write_stdout(row)
        if self._txt_path is not None:
            self._append_txt(row)
        if self._csv_path is not None:
            self._append_csv(row)

        for rm in self._means.values():
            rm.reset()
        return dict(row)

    # ----- helpers ------------------------------------------------------

    def info(self, msg: str) -> None:
        if self.is_main_process:
            line = f"[*] {msg}"
            print(line, flush=True)
            if self._txt_path is not None:
                with open(self._txt_path, "a") as f:
                    f.write(line + "\n")

    def _step_rate(self, step: int) -> Optional[float]:
        now = time.time()
        rate = None
        if self._last_dump_t is not None and self._last_dump_step is not None:
            dt = now - self._last_dump_t
            ds = step - self._last_dump_step
            if dt > 0 and ds > 0:
                rate = ds / dt
        self._last_dump_t = now
        self._last_dump_step = step
        return rate

    @staticmethod
    def _fmt(v: float) -> str:
        if abs(v) >= 1e4 or (v != 0 and abs(v) < 1e-3):
            return f"{v:.3e}"
        return f"{v:.4f}"

    def _write_stdout(self, row: Dict[str, float]) -> None:
        parts = []
        for k, v in row.items():
            if k == "step":
                parts.append(f"step {int(v)}")
            else:
                parts.append(f"{k}={self._fmt(v)}")
        print(" | ".join(parts), flush=True)

    def _append_txt(self, row: Dict[str, float]) -> None:
        with open(self._txt_path, "a") as f:
            f.write(" | ".join(f"{k}={v}" for k, v in row.items()) + "\n")

    def _append_csv(self, row: Dict[str, float]) -> None:
        write_header = not os.path.exists(self._csv_path)
        # Fix column order on first write; later keys not in header are dropped
        # so the CSV stays rectangular. Re-init the run if you add new metrics.
        if self._csv_fieldnames is None:
            self._csv_fieldnames = list(row.keys())
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in self._csv_fieldnames})


# ---------------------------------------------------------------------------
# Optional: module-level convenience matching guided-diffusion's API.
# A trainer can either instantiate `Logger` directly or call configure() once
# and use the bare functions below.
# ---------------------------------------------------------------------------

_default: Optional[Logger] = None


def configure(
    log_dir: Optional[str] = None,
    is_main_process: bool = True,
) -> Logger:
    global _default
    _default = Logger(log_dir=log_dir, is_main_process=is_main_process)
    return _default


def _ensure() -> Logger:
    global _default
    if _default is None:
        _default = Logger(log_dir=None, is_main_process=True)
    return _default


def logkv(key: str, value: Any) -> None:
    _ensure().logkv(key, value)


def logkv_mean(key: str, value: Any) -> None:
    _ensure().logkv_mean(key, value)


def dumpkvs(step: int) -> Dict[str, float]:
    return _ensure().dumpkvs(step)


def info(msg: str) -> None:
    _ensure().info(msg)
