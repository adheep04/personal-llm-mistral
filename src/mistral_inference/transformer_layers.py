from functools import partial
from typing import Optional, Tuple, Type, Union

import torch
from torch import nn
from xformers.ops.fmha import memory_efficient_attention  # type: ignore
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

from mistral_inference.args import LoraArgs
from mistral_inference.cache import CacheView
from mistral_inference.lora import LoRALinear
from mistral_inference.moe import MoeArgs, MoeLayer
from mistral_inference.rope import apply_rotary_emb


from mistral_inference.hqq import _HQQ

def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        n_kv_heads: int,
        quant_config: dict,
        lora: Optional[LoraArgs] = None,
    ):
        super().__init__()

        self.n_heads: int = n_heads
        self.head_dim: int = head_dim
        self.n_kv_heads: int = n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.head_dim**-0.5
        self._hqq = _HQQ(quant_config)
        
        if lora is None:
            self.wq = self._hqq.linear(dim, n_heads * head_dim, bias=False)
            self.wk = self._hqq.linear(dim, n_kv_heads * head_dim, bias=False)
            self.wv = self._hqq.linear(dim, n_kv_heads * head_dim, bias=False)
            self.wo = self._hqq.linear(n_heads * head_dim, dim, bias=False)
        else:
            self.wq = LoRALinear(dim, n_heads * head_dim, bias=False, rank=lora.rank, scaling=lora.scaling)
            self.wk = LoRALinear(dim, n_kv_heads * head_dim, bias=False, rank=lora.rank, scaling=lora.scaling)
            self.wv = LoRALinear(dim, n_kv_heads * head_dim, bias=False, rank=lora.rank, scaling=lora.scaling)
            self.wo = LoRALinear(n_heads * head_dim, dim, bias=False, rank=lora.rank, scaling=lora.scaling)
            
        
        
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache: Optional[CacheView] = None,
        mask: Optional[BlockDiagonalMask] = None,
    ) -> torch.Tensor:
        assert mask is None or cache is None
        seqlen_sum, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(seqlen_sum, self.n_heads, self.head_dim)
        xk = xk.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        xv = xv.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if cache is None:
            key, val = xk, xv
        elif cache.prefill:
            key, val = cache.interleave_kv(xk, xv)
            cache.update(xk, xv)
        else:
            cache.update(xk, xv)
            key, val = cache.key, cache.value
            key = key.view(seqlen_sum * cache.max_seq_len, self.n_kv_heads, self.head_dim)
            val = val.view(seqlen_sum * cache.max_seq_len, self.n_kv_heads, self.head_dim)

        # Repeat keys and values to match number of query heads
        key, val = repeat_kv(key, val, self.repeats, dim=1)

        # xformers requires (B=1, S, H, D)
        xq, key, val = xq[None, ...], key[None, ...], val[None, ...]
        output = memory_efficient_attention(xq, key, val, mask if cache is None else cache.mask)
        output = output.view(seqlen_sum, self.n_heads * self.head_dim)

        assert isinstance(output, torch.Tensor)

        return self.wo(output)  # type: ignore


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, quant_config: any, lora: Optional[LoraArgs] = None):
        super().__init__()
        
        self._hqq = _HQQ(quant_config)

        if lora is None:
            self.w1 = nn.Linear(dim, hidden_dim, bias=False)
            self.w2 = nn.Linear(hidden_dim, dim, bias=False)
            self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        else:
            self.w1 = self._hqq.linear(dim, hidden_dim, bias=False)
            self.w2 = self._hqq.linear(hidden_dim, dim, bias=False)
            self.w3 = self._hqq.linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))  # type: ignore


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        norm_eps: float,
        quant_config: dict,
        lora: Optional[LoraArgs] = None,
        moe: Optional[MoeArgs] = None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.attention = Attention(
            dim=dim,
            n_heads=n_heads,
            head_dim=head_dim,
            n_kv_heads=n_kv_heads,
            lora=lora,
            quant_config=quant_config
        )
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

        self.feed_forward: nn.Module
        if moe is not None:
            self.feed_forward = MoeLayer(
                experts=[FeedForward(dim=dim, hidden_dim=hidden_dim, lora=lora) for _ in range(moe.num_experts)],
                gate=nn.Linear(dim, moe.num_experts, bias=False),
                moe_args=moe,
            )
        else:
            self.feed_forward = FeedForward(dim=dim, hidden_dim=hidden_dim, lora=lora, quant_config=quant_config)
            
        self._hqq = _HQQ(quant_config)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache: Optional[CacheView] = None,
        mask: Optional[BlockDiagonalMask] = None,
    ) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x), freqs_cis, cache)
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out
