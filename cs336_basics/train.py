from typing import BinaryIO, IO

import os
import torch

from .training.checkpoint import load_checkpoint as _load_checkpoint_with_extra
from .training.checkpoint import save_checkpoint as _save_checkpoint_with_extra
from .training.data import get_batch
from .training.decode import decode, sample_next_token
from .training.logging import MetricSnapshot, SmoothedMetric, TrainLogger, create_train_logger
from .training.loop import evaluate_loop, train_loop
from .training.losses import cross_entropy
from .training.optim import AdamW, get_lr_cosine_schedule, gradient_clipping


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    _save_checkpoint_with_extra(
        model=model,
        optimizer=optimizer,
        iteration=iteration,
        out=out,
        extra_state=None,
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    iteration, _ = _load_checkpoint_with_extra(
        src=src,
        model=model,
        optimizer=optimizer,
    )
    return iteration


__all__ = [
    "AdamW",
    "MetricSnapshot",
    "SmoothedMetric",
    "TrainLogger",
    "create_train_logger",
    "cross_entropy",
    "decode",
    "evaluate_loop",
    "get_batch",
    "get_lr_cosine_schedule",
    "gradient_clipping",
    "load_checkpoint",
    "sample_next_token",
    "save_checkpoint",
    "train_loop",
]

