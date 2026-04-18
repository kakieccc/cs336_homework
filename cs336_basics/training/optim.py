import math
from typing import Iterable, Optional

import torch


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    torch.nn.utils.clip_grad_norm_(parameters, max_l2_norm)


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss: Optional[torch.Tensor] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                step_t = state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - (beta1**step_t)
                bias_correction2 = 1.0 - (beta2**step_t)
                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        if warmup_iters == 0:
            return max_learning_rate
        return max_learning_rate * (it / warmup_iters)

    if it <= warmup_iters:
        return max_learning_rate

    if it >= cosine_cycle_iters:
        return min_learning_rate

    decay_iters = cosine_cycle_iters - warmup_iters
    if decay_iters <= 0:
        return min_learning_rate

    decay_ratio = (it - warmup_iters) / decay_iters
    cosine_coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + cosine_coeff * (max_learning_rate - min_learning_rate)
