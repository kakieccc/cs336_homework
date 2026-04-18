import torch


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if inputs.ndim != 2:
        raise ValueError("inputs must have shape (batch_size, vocab_size)")
    if targets.ndim != 1:
        raise ValueError("targets must have shape (batch_size,)")
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError("inputs and targets must have the same batch size")

    logsumexp = torch.logsumexp(inputs, dim=-1)
    target_logits = inputs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (logsumexp - target_logits).mean()
