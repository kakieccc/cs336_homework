import torch


@torch.no_grad()
def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_q: float | None = None,
) -> torch.Tensor:
    if top_q is not None:
        top_p = top_q

    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if not (0 < top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1]")

    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, vocab_size) or (vocab_size,)")

    scaled_logits = logits / temperature

    if top_p >= 1.0:
        probs = torch.softmax(scaled_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    sorted_logits, sorted_indices = torch.sort(scaled_logits, dim=-1, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    keep_mask = (cum_probs - sorted_probs) < top_p
    keep_mask[..., 0] = True

    filtered_sorted_logits = sorted_logits.masked_fill(~keep_mask, -torch.inf)
    filtered_sorted_probs = torch.softmax(filtered_sorted_logits, dim=-1)

    next_sorted_idx = torch.multinomial(filtered_sorted_probs, num_samples=1)
    return sorted_indices.gather(dim=-1, index=next_sorted_idx).squeeze(-1)


@torch.no_grad()
def decode(
    model: torch.nn.Module,
    prefix: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_q: float | None = None,
) -> torch.Tensor:
    if prefix.ndim != 2:
        raise ValueError("prefix must have shape (batch, seq)")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")

    generated = prefix
    context_length = getattr(model, "context_length", None)

    for _ in range(max_new_tokens):
        model_input = generated[:, -context_length:] if context_length is not None else generated
        logits = model(model_input)
        next_logits = logits[:, -1, :]
        next_token = sample_next_token(
            next_logits,
            temperature=temperature,
            top_p=top_p,
            top_q=top_q,
        )
        generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

    return generated
