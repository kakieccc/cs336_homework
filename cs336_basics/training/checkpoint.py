import os
from typing import BinaryIO, IO

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
    extra_state: dict | None = None,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": int(iteration),
    }
    if extra_state is not None:
        payload["extra_state"] = extra_state
    torch.save(payload, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    map_location=None,
) -> tuple[int, dict | None]:
    checkpoint = torch.load(src, map_location=map_location)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    iteration = int(checkpoint["iteration"])
    extra_state = checkpoint.get("extra_state")
    return iteration, extra_state
