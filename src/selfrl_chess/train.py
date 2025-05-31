"""Training loop.

Highlights:

* AMP (mixed precision) on CUDA via ``torch.amp``.
* Cosine learning-rate schedule with linear warm-up.
* Adam-W with weight decay.
* Gradient clipping.
* EMA shadow network.
* Tensorboard logging if ``tensorboard`` is installed.
* Atomic checkpoint writes.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .config import (
    BATCH_SIZE,
    DEVICE,
    EMA_DECAY,
    EPOCHS,
    GRAD_CLIP,
    LEARNING_RATE,
    LOG_DIR,
    MODELS_DIR,
    SEED,
    WEIGHT_DECAY,
)
from .ema import ModelEMA
from .network import PolicyValueNet
from .utils import configure_logging, save_checkpoint, seed_everything

log = logging.getLogger(__name__)


class ShardDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Streaming view over one HDF5 shard.

    The HDF5 file is opened lazily per worker so that ``DataLoader``
    multiprocessing keeps working without pickling open handles.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        with h5py.File(self.path, "r") as f:
            self._length = int(f["states"].shape[0])
        self._handle: h5py.File | None = None

    def _file(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f = self._file()
        state = torch.from_numpy(f["states"][idx]).float()
        policy = torch.from_numpy(f["policies"][idx]).float()
        value = torch.tensor(float(f["values"][idx]))
        return state, policy, value


@dataclass
class TrainConfig:
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    grad_clip: float = GRAD_CLIP
    ema_decay: float = EMA_DECAY
    warmup_steps: int = 1_000
    seed: int = SEED
    device: str = DEVICE


def _cosine_lr(step: int, *, base_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _make_loader(shards: Iterable[Path], batch_size: int) -> DataLoader:
    datasets = [ShardDataset(p) for p in shards]
    if not datasets:
        raise FileNotFoundError("no replay shards available for training")
    ds = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def train(
    shards: Iterable[Path],
    *,
    cfg: TrainConfig | None = None,
    model: PolicyValueNet | None = None,
) -> Path:
    """Train ``model`` on ``shards``. Returns the path of the best checkpoint."""
    cfg = cfg or TrainConfig()
    configure_logging()
    seed_everything(cfg.seed)

    shards = list(shards)
    loader = _make_loader(shards, cfg.batch_size)
    total_steps = cfg.epochs * len(loader)

    device = torch.device(cfg.device)
    model = model or PolicyValueNet()
    model = model.to(device)
    ema = ModelEMA(model, decay=cfg.ema_decay)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    writer = _maybe_tensorboard()

    best_loss = math.inf
    best_path = MODELS_DIR / "best_model.pth"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_p = 0.0
        epoch_v = 0.0
        n = 0
        for states, policies, values in loader:
            states = states.to(device, non_blocking=True)
            policies = policies.to(device, non_blocking=True)
            values = values.to(device, non_blocking=True)

            lr = _cosine_lr(
                step,
                base_lr=cfg.learning_rate,
                warmup=cfg.warmup_steps,
                total=total_steps,
            )
            for g in optimizer.param_groups:
                g["lr"] = lr

            ctx = torch.amp.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
            with ctx:
                log_p, pred_v = model(states)
                pred_v = pred_v.view(-1)
                loss_v = F.mse_loss(pred_v, values)
                loss_p = -(policies * log_p).sum(dim=1).mean()
                loss = loss_v + loss_p

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            bs = states.size(0)
            epoch_loss += loss.item() * bs
            epoch_p += loss_p.item() * bs
            epoch_v += loss_v.item() * bs
            n += bs
            step += 1

            if writer is not None and step % 50 == 0:
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/loss_policy", loss_p.item(), step)
                writer.add_scalar("train/loss_value", loss_v.item(), step)
                writer.add_scalar("train/lr", lr, step)

        avg = epoch_loss / max(1, n)
        log.info(
            "epoch %d/%d loss=%.4f policy=%.4f value=%.4f",
            epoch,
            cfg.epochs,
            avg,
            epoch_p / max(1, n),
            epoch_v / max(1, n),
        )

        ckpt = MODELS_DIR / f"checkpoint_epoch{epoch}.pth"
        save_checkpoint(
            ckpt,
            model=model,
            optimizer=optimizer,
            metadata={"epoch": epoch, "loss": avg, "ema_decay": cfg.ema_decay},
        )
        save_checkpoint(
            MODELS_DIR / f"ema_epoch{epoch}.pth",
            model=ema.module,
            metadata={"epoch": epoch, "ema_decay": cfg.ema_decay},
        )
        if avg < best_loss:
            best_loss = avg
            save_checkpoint(best_path, model=ema.module, metadata={"epoch": epoch, "loss": avg})

    log.info("training done; best loss=%.4f -> %s", best_loss, best_path)
    if writer is not None:
        writer.close()
    return best_path


def _maybe_tensorboard():
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:  # pragma: no cover - optional dependency
        return None
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(LOG_DIR))
