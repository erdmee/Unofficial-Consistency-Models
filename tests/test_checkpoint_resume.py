"""Tests for wandb_run_id persistence and off-by-one resume semantics."""
import torch
import torch.nn as nn
from torch.optim import SGD

from cm.utils.checkpoint import save_checkpoint, load_checkpoint


def _build_pair(tmp_path):
    """Tiny model + optimizer + EMA copies for round-trip tests."""
    model = nn.Linear(4, 4)
    ema = nn.Linear(4, 4)
    sampling_ema = nn.Linear(4, 4)
    opt = SGD(model.parameters(), lr=0.1)
    # one fake step so optimizer has state
    model(torch.randn(2, 4)).sum().backward()
    opt.step()
    return model, ema, sampling_ema, opt


def test_save_load_roundtrip_with_wandb_run_id(tmp_path):
    model, ema, sampling_ema, opt = _build_pair(tmp_path)
    path = tmp_path / "step_000100.pt"
    save_checkpoint(
        save_path=str(path),
        step=100,
        ema_model=ema,
        model=model,
        optimizer=opt,
        sampling_ema_model=sampling_ema,
        wandb_run_id="abc123def",
    )

    model2 = nn.Linear(4, 4)
    ema2 = nn.Linear(4, 4)
    sampling_ema2 = nn.Linear(4, 4)
    opt2 = SGD(model2.parameters(), lr=0.1)
    step, _, run_id = load_checkpoint(
        load_path=str(path),
        ema_model=ema2,
        model=model2,
        optimizer=opt2,
        sampling_ema_model=sampling_ema2,
    )
    assert step == 100
    assert run_id == "abc123def"


def test_load_returns_none_run_id_when_not_saved(tmp_path):
    model, ema, sampling_ema, opt = _build_pair(tmp_path)
    path = tmp_path / "step_000050.pt"
    save_checkpoint(
        save_path=str(path),
        step=50,
        ema_model=ema,
        model=model,
        optimizer=opt,
        sampling_ema_model=sampling_ema,
        # wandb_run_id intentionally omitted → defaults to None
    )

    ema2 = nn.Linear(4, 4)
    _, _, run_id = load_checkpoint(
        load_path=str(path),
        ema_model=ema2,
    )
    assert run_id is None


def test_legacy_checkpoint_without_wandb_run_id_key(tmp_path):
    """A ckpt saved before this field existed should still load cleanly."""
    path = tmp_path / "legacy.pt"
    legacy = {
        "step": 42,
        "ema_state_dict": nn.Linear(4, 4).state_dict(),
        "model_state_dict": nn.Linear(4, 4).state_dict(),
        "optimizer_state_dict": {},
        "config": {"foo": "bar"},
        # NOTE: no "wandb_run_id" key at all
    }
    torch.save(legacy, path)

    ema2 = nn.Linear(4, 4)
    step, cfg, run_id = load_checkpoint(load_path=str(path), ema_model=ema2)
    assert step == 42
    assert cfg == {"foo": "bar"}
    assert run_id is None


def test_scaler_state_roundtrip(tmp_path):
    """A non-empty scaler state must survive save → load."""
    model, ema, sampling_ema, opt = _build_pair(tmp_path)
    path = tmp_path / "step_000010.pt"
    save_checkpoint(
        save_path=str(path),
        step=10,
        ema_model=ema,
        model=model,
        optimizer=opt,
        sampling_ema_model=sampling_ema,
    )
    # Inject a synthetic fp16-style scaler state (CPU tests can't run a real CUDA scaler)
    ckpt = torch.load(path)
    ckpt["scaler_state_dict"] = {
        "scale": 1024.0,
        "growth_factor": 2.0,
        "backoff_factor": 0.5,
        "growth_interval": 2000,
        "_growth_tracker": 7,
    }
    torch.save(ckpt, path)

    scaler = torch.amp.GradScaler("cpu", enabled=True)
    load_checkpoint(load_path=str(path), ema_model=nn.Linear(4, 4), scaler=scaler)
    assert scaler.get_scale() == 1024.0


def test_scaler_empty_state_is_skipped(tmp_path):
    """A disabled scaler saves {} — loading must skip it instead of raising."""
    model, ema, sampling_ema, opt = _build_pair(tmp_path)
    path = tmp_path / "step_000011.pt"
    save_checkpoint(
        save_path=str(path),
        step=11,
        ema_model=ema,
        model=model,
        optimizer=opt,
        sampling_ema_model=sampling_ema,
        scaler=torch.amp.GradScaler("cpu", enabled=False),  # state_dict() == {}
    )
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    load_checkpoint(load_path=str(path), ema_model=nn.Linear(4, 4), scaler=scaler)  # no raise


def test_legacy_checkpoint_without_scaler_key(tmp_path):
    """Checkpoints saved before scaler_state_dict existed must still load."""
    path = tmp_path / "legacy_noscaler.pt"
    legacy = {
        "step": 7,
        "ema_state_dict": nn.Linear(4, 4).state_dict(),
        # NOTE: no "scaler_state_dict" key at all
    }
    torch.save(legacy, path)
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    step, _, _ = load_checkpoint(load_path=str(path), ema_model=nn.Linear(4, 4), scaler=scaler)
    assert step == 7


def test_resume_start_step_is_saved_plus_one(tmp_path, monkeypatch):
    """Off-by-one fix: a ckpt saved at step N must resume at step N+1, not N."""
    # Import here so monkeypatching cm.training.train doesn't leak between tests
    from cm.training import ct_trainer as ct_mod

    model, ema, sampling_ema, opt = _build_pair(tmp_path)
    ckpt_path = tmp_path / "step_001234.pt"
    save_checkpoint(
        save_path=str(ckpt_path),
        step=1234,
        ema_model=ema,
        model=model,
        optimizer=opt,
        sampling_ema_model=sampling_ema,
        wandb_run_id="run-xyz",
    )

    # Build a minimal stand-in for CTTrainer.__init__ resume path without
    # actually constructing the trainer (which requires data, lpips, etc.)
    class _Stub:
        is_distributed = False
        is_main_process = True
        device = torch.device("cpu")

        def __init__(self, ckpt_path, ema, model, sampling_ema, opt):
            self.target_model = ema
            self.online_model = model
            self.sampling_ema_model = sampling_ema
            self.optimizer = opt
            self.scaler = torch.amp.GradScaler("cpu", enabled=False)
            self.start_step = 0
            self.wandb_run_id = None
            ct_mod.CTTrainer._resume_training(self, str(ckpt_path))

    stub = _Stub(
        ckpt_path,
        nn.Linear(4, 4),
        nn.Linear(4, 4),
        nn.Linear(4, 4),
        SGD(nn.Linear(4, 4).parameters(), lr=0.1),
    )
    assert stub.start_step == 1235, f"expected 1235 (saved+1), got {stub.start_step}"
    assert stub.wandb_run_id == "run-xyz"
