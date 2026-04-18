from collections.abc import Callable
import time

import numpy as np
import torch

from .data import get_batch
from .losses import cross_entropy
from .logging import TrainLogger


def _compute_grad_l2_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        grad = p.grad.detach()
        total += float(torch.sum(grad * grad).item())
    return total ** 0.5


def evaluate_loop(
    model: torch.nn.Module,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    eval_steps: int,
) -> dict[str, float]:
    model_was_training = model.training
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for _ in range(eval_steps):
            x, y = get_batch(dataset, batch_size=batch_size, context_length=context_length, device=device)
            logits = model(x)
            loss = cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += float(loss.item())
            total_tokens += int(y.numel())

    if model_was_training:
        model.train()

    mean_loss = total_loss / max(1, eval_steps)
    perplexity = float(torch.exp(torch.tensor(mean_loss)).item())
    return {
        "loss": mean_loss,
        "perplexity": perplexity,
        "tokens": float(total_tokens),
    }


def train_loop(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataset: np.ndarray,
    val_dataset: np.ndarray | None,
    batch_size: int,
    context_length: int,
    max_steps: int,
    device: str,
    logger: TrainLogger | None = None,
    lr_schedule_fn: Callable[[int], float] | None = None,
    grad_clip_norm: float | None = None,
    eval_interval: int = 0,
    eval_steps: int = 10,
) -> dict[str, float | int]:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    model.to(device)
    model.train()

    start_time = time.time()
    total_tokens = 0
    final_loss = 0.0

    for step in range(max_steps):
        if lr_schedule_fn is not None:
            lr = float(lr_schedule_fn(step))
            for group in optimizer.param_groups:
                group["lr"] = lr
        else:
            lr = float(optimizer.param_groups[0]["lr"])

        x, y = get_batch(train_dataset, batch_size=batch_size, context_length=context_length, device=device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()

        grad_norm = _compute_grad_l2_norm(model.parameters())
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        final_loss = float(loss.item())
        tokens_this_step = int(y.numel())
        total_tokens += tokens_this_step
        elapsed = max(time.time() - start_time, 1e-12)
        tokens_per_sec = total_tokens / elapsed

        if logger is not None:
            logger.log_step(
                step=step,
                prefix="train",
                metrics={
                    "loss": final_loss,
                    "lr": lr,
                    "grad_norm": grad_norm,
                    "tokens_per_sec": tokens_per_sec,
                },
            )

        if (
            logger is not None
            and val_dataset is not None
            and eval_interval > 0
            and ((step + 1) % eval_interval == 0)
        ):
            eval_metrics = evaluate_loop(
                model=model,
                dataset=val_dataset,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
                eval_steps=eval_steps,
            )
            logger.log_step(step=step, prefix="val", metrics=eval_metrics)

    total_time = time.time() - start_time
    return {
        "final_train_loss": final_loss,
        "steps": int(max_steps),
        "tokens": int(total_tokens),
        "total_time_s": float(total_time),
        "tokens_per_s": float(total_tokens / max(total_time, 1e-12)),
    }
