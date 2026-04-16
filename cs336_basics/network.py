import math
from typing import Optional

import torch
import torch.nn as nn

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = torch.amax(x, dim=dim, keepdim=True)
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)
    


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, device=None, dtype=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weight = nn.Parameter(
            torch.empty((vocab_size, d_model), device=device, dtype=dtype)
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones((d_model,), device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        rmsnorm = (x / rms) * self.weight
        return rmsnorm.to(in_dtype)


class SwiGLU:
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = silu(self.w1.forward(x))
        b = self.w3.forward(x)
        return self.w2.forward(a * b)


def scaled_dot_product_attention(
    Q: torch.Tensor,  # (..., q_len, d_k)
    K: torch.Tensor,  # (..., k_len, d_k)
    V: torch.Tensor,  # (..., k_len, d_v)
    mask: Optional[torch.Tensor] = None,  # (..., q_len, k_len), bool
) -> torch.Tensor:
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, -torch.finfo(scores.dtype).max)
    probs = softmax(scores, dim=-1)
    return probs @ V


class RoPE:
    def __init__(self, d_k: int, theta: float = 10000.0, max_seq_len: int = 2048, device=None):
        if d_k % 2 != 0:
            raise ValueError("RoPE requires even d_k.")
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.device = device

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x_fp32 = x.to(torch.float32)

        half = self.d_k // 2
        i = torch.arange(half, device=x.device, dtype=torch.float32)
        inv_freq = self.theta ** (-2.0 * i / self.d_k)  # (half,)

        # 显式把 token_positions reshape 成可广播到 x 的形状：
        # target: (*prefix_dims, seq_len, 1), 其中 prefix_dims 数量 = x.ndim - 2
        prefix_ndim = x.ndim - 2
        pos_prefix_ndim = token_positions.ndim - 1
        num_missing = prefix_ndim - pos_prefix_ndim
        
        pos_shape = token_positions.shape[:-1] + (1,) * num_missing + (token_positions.shape[-1], 1)
        pos = token_positions.to(device=x.device, dtype=torch.float32).reshape(pos_shape)

        angles = pos * inv_freq  # (*prefix_dims, seq_len, half)
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        # 把最后一维按 pair 拆开做旋转
        x_pair = x_fp32.reshape(*x_fp32.shape[:-1], half, 2)
        x0 = x_pair[..., 0]
        x1 = x_pair[..., 1]

        y0 = x0 * cos - x1 * sin
        y1 = x0 * sin + x1 * cos

        out = torch.stack((y0, y1), dim=-1).reshape_as(x_fp32)
        return out.to(in_dtype)


class MultiHeadSelfAttention:
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        max_seq_len: int = 2048,
        device=None,
        dtype=None,
    ):
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.use_rope = use_rope
        self.rope = RoPE(self.d_head, theta=rope_theta, max_seq_len=max_seq_len, device=device) if use_rope else None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, s, d_model) -> (b, h, s, d_head)
        b, s, _ = x.shape
        x = x.view(b, s, self.num_heads, self.d_head)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, h, s, d_head) -> (b, s, d_model)
        b, h, s, d = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(b, s, h * d)

    def forward(self, x: torch.Tensor, token_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (b, s, d_model)
        q = self._split_heads(self.q_proj.forward(x))
        k = self._split_heads(self.k_proj.forward(x))
        v = self._split_heads(self.v_proj.forward(x))

        if self.use_rope:
            # TODO: 对 q 和 k 应用 RoPE；v 不做 RoPE
            if token_positions is None:
                raise ValueError("token_positions is required when use_rope=True.")
            q = self.rope.forward(q, token_positions)
            k = self.rope.forward(k, token_positions)

        # 因果 mask: (1, 1, s, s)
        s = x.shape[1]
        causal = torch.tril(torch.ones((s, s), dtype=torch.bool, device=x.device)).view(1, 1, s, s)

        attn_out = scaled_dot_product_attention(q, k, v, mask=causal)
        out = self._merge_heads(attn_out)
        return self.output_proj.forward(out)


class TransformerBlock:
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float = 10000.0,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        self.ln1 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            use_rope=True,
            rope_theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # Pre-Norm
        # TODO: x = x + attn(ln1(x)); x = x + ffn(ln2(x))
        raise NotImplementedError


class TransformerLM:
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = [
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                eps=eps,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ]
        self.ln_final = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (b, s)
        b, s = input_ids.shape
        if s > self.context_length:
            raise ValueError("sequence length exceeds context_length.")

        x = self.token_embeddings.forward(input_ids)  # (b, s, d_model)

        # positions: (b, s)
        positions = torch.arange(s, device=input_ids.device).unsqueeze(0).expand(b, s)

        for block in self.layers:
            x = block.forward(x, token_positions=positions)

        x = self.ln_final.forward(x)
        logits = self.lm_head.forward(x)  # (b, s, vocab_size)
        return logits