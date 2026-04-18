import numpy as np
import torch


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if dataset.ndim != 1:
        raise ValueError("dataset must be a 1D numpy array")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if context_length <= 0:
        raise ValueError("context_length must be positive")

    num_possible_starts = dataset.shape[0] - context_length
    if num_possible_starts <= 0:
        raise ValueError("dataset is too short for the requested context_length")

    starts = torch.randint(0, num_possible_starts, (batch_size,))
    dataset_tensor = torch.as_tensor(dataset, dtype=torch.long)

    x = torch.stack([dataset_tensor[s : s + context_length] for s in starts.tolist()])
    y = torch.stack([dataset_tensor[s + 1 : s + context_length + 1] for s in starts.tolist()])
    return x.to(device), y.to(device)
