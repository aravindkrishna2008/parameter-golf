from __future__ import annotations

import copy
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from prepare import (
    DistributedTokenLoader,
    RuntimeConfig,
    ValidationLUTs,
    append_result_row,
    artifact_bytes_for_blob,
    build_experiment_metadata,
    build_runtime_config,
    build_sentencepiece_luts,
    count_code_bytes,
    decompress_quantized_payload,
    default_quantization_spec,
    dequantize_state_dict,
    eval_val,
    load_tokenizer,
    load_validation_tokens,
    quantize_state_dict,
    stable_hash,
    summarize_runtime,
)


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    iterations: int
    warmdown_iters: int
    warmup_steps: int
    train_batch_tokens: int
    qk_gain_init: float
    num_layers: int
    model_dim: int
    num_heads: int
    num_kv_heads: int
    mlp_mult: float
    tie_embeddings: bool
    rope_base: float
    logit_softcap: float
    embed_lr: float
    head_lr: float
    tied_embed_lr: float
    tied_embed_init_std: float
    matrix_lr: float
    scalar_lr: float
    muon_momentum: float
    muon_backend_steps: int
    muon_momentum_warmup_start: float
    muon_momentum_warmup_steps: int
    beta1: float
    beta2: float
    adam_eps: float
    grad_clip_norm: float
    calibration_source: str
    calibration_batches: int
    calibration_temperature: float
    enable_bigram_hash: bool
    bigram_vocab_size: int
    bigram_dim: int
    xsa_last_n: int
    rope_dims: int
    late_qat_threshold: float
    openelm_style: bool
    openelm_ffn_min_mult: float
    openelm_ffn_max_mult: float
    openelm_use_swiglu: bool

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def build_train_config(runtime: RuntimeConfig) -> TrainConfig:
    train_batch_tokens_default = 131_072 if runtime.is_proxy else 786_432
    warmdown_default = 1_000 if runtime.is_proxy else 3_500
    return TrainConfig(
        seed=int(os.environ.get("SEED", 1337)),
        iterations=int(os.environ.get("ITERATIONS", 20_000)),
        warmdown_iters=int(os.environ.get("WARMDOWN_ITERS", warmdown_default)),
        warmup_steps=int(os.environ.get("WARMUP_STEPS", 20)),
        train_batch_tokens=int(os.environ.get("TRAIN_BATCH_TOKENS", train_batch_tokens_default)),
        qk_gain_init=float(os.environ.get("QK_GAIN_INIT", 1.5)),
        num_layers=int(os.environ.get("NUM_LAYERS", 11)),
        model_dim=int(os.environ.get("MODEL_DIM", 512)),
        num_heads=int(os.environ.get("NUM_HEADS", 8)),
        num_kv_heads=int(os.environ.get("NUM_KV_HEADS", 4)),
        mlp_mult=float(os.environ.get("MLP_MULT", 3.0)),
        tie_embeddings=bool(int(os.environ.get("TIE_EMBEDDINGS", "1"))),
        rope_base=float(os.environ.get("ROPE_BASE", 10000.0)),
        logit_softcap=float(os.environ.get("LOGIT_SOFTCAP", 30.0)),
        embed_lr=float(os.environ.get("EMBED_LR", 0.6)),
        head_lr=float(os.environ.get("HEAD_LR", 0.008)),
        tied_embed_lr=float(os.environ.get("TIED_EMBED_LR", 0.035)),
        tied_embed_init_std=float(os.environ.get("TIED_EMBED_INIT_STD", 0.005)),
        matrix_lr=float(os.environ.get("MATRIX_LR", 0.025)),
        scalar_lr=float(os.environ.get("SCALAR_LR", 0.025)),
        muon_momentum=float(os.environ.get("MUON_MOMENTUM", 0.99)),
        muon_backend_steps=int(os.environ.get("MUON_BACKEND_STEPS", 5)),
        muon_momentum_warmup_start=float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92)),
        muon_momentum_warmup_steps=int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1_500)),
        beta1=float(os.environ.get("BETA1", 0.9)),
        beta2=float(os.environ.get("BETA2", 0.95)),
        adam_eps=float(os.environ.get("ADAM_EPS", 1e-8)),
        grad_clip_norm=float(os.environ.get("GRAD_CLIP_NORM", 0.3)),
        calibration_source=os.environ.get("CALIBRATION_SOURCE", "train_stream"),
        calibration_batches=int(os.environ.get("CALIBRATION_BATCHES", 8)),
        calibration_temperature=float(os.environ.get("CALIBRATION_TEMPERATURE", 0.0)),
        enable_bigram_hash=bool(int(os.environ.get("ENABLE_BIGRAM_HASH", "0"))),
        bigram_vocab_size=int(os.environ.get("BIGRAM_VOCAB_SIZE", 3072)),
        bigram_dim=int(os.environ.get("BIGRAM_DIM", 112)),
        xsa_last_n=int(os.environ.get("XSA_LAST_N", 0)),
        rope_dims=int(os.environ.get("ROPE_DIMS", 0)),
        late_qat_threshold=float(os.environ.get("LATE_QAT_THRESHOLD", 0.0)),
        openelm_style=bool(int(os.environ.get("OPENELM_STYLE", "0"))),
        openelm_ffn_min_mult=float(os.environ.get("OPENELM_FFN_MIN_MULT", 2.5)),
        openelm_ffn_max_mult=float(os.environ.get("OPENELM_FFN_MAX_MULT", 3.5)),
        openelm_use_swiglu=bool(int(os.environ.get("OPENELM_USE_SWIGLU", "0"))),
    )


def zeropower_via_newtonschulz5(grad: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = grad.bfloat16()
    x /= x.norm() + eps
    transposed = grad.size(0) > grad.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        a_t = x @ x.T
        b_t = b * a_t + c * a_t @ a_t
        x = a * x + b_t @ x
    return x.T if transposed else x


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, param in enumerate(params):
                if i % world_size == rank and param.grad is not None:
                    grad = param.grad
                    state = self.state[param]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)
                    if nesterov:
                        grad = grad.add(buf, alpha=momentum)
                    grad = zeropower_via_newtonschulz5(grad, steps=backend_steps)
                    grad *= max(1, grad.size(0) / grad.size(1)) ** 0.5
                    updates_flat[curr : curr + param.numel()] = grad.reshape(-1)
                curr += param.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for param in params:
                grad = updates_flat[curr : curr + param.numel()].view_as(param).to(dtype=param.dtype)
                param.add_(grad, alpha=-lr)
                curr += param.numel()

        return loss


def _fake_quantize_rowwise(weight: Tensor, qmax: int) -> Tensor:
    if weight.ndim != 2:
        return weight
    clip_abs = weight.abs().amax(dim=1).clamp_min(1.0 / qmax)
    scale = clip_abs / qmax
    quantized = (weight / scale[:, None]).round().clamp(-qmax, qmax)
    dequantized = quantized * scale[:, None]
    return weight + (dequantized - weight).detach()


class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    _qat_enabled = False
    _qat_qmax = 31

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight.to(x.dtype)
        if CastedLinear._qat_enabled and self.training and self.weight.ndim == 2:
            weight = _fake_quantize_rowwise(weight, CastedLinear._qat_qmax)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)


def restore_low_dim_params_to_fp32(module: nn.Module, control_patterns: tuple[str, ...]) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in control_patterns)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, rope_dims: int = 0):
        super().__init__()
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        if self.rope_dims % 2 != 0:
            raise ValueError(f"rope_dims must be even, got {self.rope_dims}")
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int = 0) -> Tensor:
    active_dims = rope_dims if rope_dims > 0 else x.size(-1)
    if active_dims < x.size(-1):
        x_rot, x_pass = x[..., :active_dims], x[..., active_dims:]
    else:
        x_rot, x_pass = x, None
    half = x_rot.size(-1) // 2
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    rotated = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
    return torch.cat((rotated, x_pass), dim=-1) if x_pass is not None else rotated


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        rope_dims: int = 0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.rope_dims = rope_dims if rope_dims > 0 else self.head_dim
        if self.rope_dims > self.head_dim:
            raise ValueError(f"ROPE_DIMS={self.rope_dims} exceeds head_dim={self.head_dim}")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base, rope_dims=self.rope_dims)
        self.use_xsa = False

    def _apply_xsa(self, y: Tensor, v: Tensor) -> Tensor:
        # y: [B, H, T, D], v: [B, Hkv, T, D]
        bsz, heads, seqlen, dim = y.shape
        kv_heads = v.size(1)
        group = heads // kv_heads
        y_t = y.transpose(1, 2).reshape(bsz, seqlen, kv_heads, group, dim)
        v_t = F.normalize(v.transpose(1, 2), dim=-1).unsqueeze(-2)
        proj = (y_t * v_t).sum(dim=-1, keepdim=True) * v_t
        return (y_t - proj).reshape(bsz, seqlen, heads, dim).transpose(1, 2).contiguous()

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        if self.use_xsa:
            y = self._apply_xsa(y, v)
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def _hash(self, tokens: Tensor) -> Tensor:
        tokens32 = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(tokens32)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * tokens32[..., 1:], 27191 * tokens32[..., :-1]) % mod
        return out.long()

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self._hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, use_swiglu: bool):
        super().__init__()
        self.use_swiglu = use_swiglu
        if use_swiglu:
            self.gate = CastedLinear(dim, hidden_dim, bias=False)
            self.value = CastedLinear(dim, hidden_dim, bias=False)
            self.proj = CastedLinear(hidden_dim, dim, bias=False)
        else:
            self.fc = CastedLinear(dim, hidden_dim, bias=False)
            self.proj = CastedLinear(hidden_dim, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        if self.use_swiglu:
            return self.proj(F.silu(self.gate(x)) * self.value(x))
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        hidden_dim: int,
        rope_base: float,
        qk_gain_init: float,
        rope_dims: int,
        use_swiglu: bool,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, rope_dims=rope_dims)
        self.mlp = FeedForward(dim, hidden_dim, use_swiglu=use_swiglu)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(self.attn_norm(x))
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


def make_layer_hidden_dims(config: TrainConfig) -> list[int]:
    if not config.openelm_style:
        return [int(config.model_dim * config.mlp_mult)] * config.num_layers
    if config.openelm_ffn_min_mult <= 0 or config.openelm_ffn_max_mult <= 0:
        raise ValueError("OpenELM multipliers must be positive")
    if config.openelm_ffn_max_mult < config.openelm_ffn_min_mult:
        raise ValueError("OPENELM_FFN_MAX_MULT must be >= OPENELM_FFN_MIN_MULT")
    if config.num_layers == 1:
        return [int(config.model_dim * config.openelm_ffn_max_mult)]
    hidden_dims = []
    for layer_idx in range(config.num_layers):
        frac = layer_idx / (config.num_layers - 1)
        mult = (1.0 - frac) * config.openelm_ffn_min_mult + frac * config.openelm_ffn_max_mult
        hidden_dims.append(int(config.model_dim * mult))
    return hidden_dims


class GPT(nn.Module):
    def __init__(self, runtime: RuntimeConfig, config: TrainConfig):
        super().__init__()
        if config.logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {config.logit_softcap}")
        self.tie_embeddings = config.tie_embeddings
        self.logit_softcap = config.logit_softcap
        self.tied_embed_init_std = config.tied_embed_init_std
        self.vocab_size = runtime.vocab_size
        self.model_dim = config.model_dim
        self.bigram = (
            BigramHashEmbedding(config.bigram_vocab_size, config.bigram_dim, config.model_dim)
            if config.enable_bigram_hash
            else None
        )
        self.tok_emb = nn.Embedding(runtime.vocab_size, config.model_dim)
        hidden_dims = make_layer_hidden_dims(config)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=config.model_dim,
                    num_heads=config.num_heads,
                    num_kv_heads=config.num_kv_heads,
                    hidden_dim=hidden_dims[layer_idx],
                    rope_base=config.rope_base,
                    qk_gain_init=config.qk_gain_init,
                    rope_dims=config.rope_dims,
                    use_swiglu=config.openelm_style and config.openelm_use_swiglu,
                )
                for layer_idx in range(config.num_layers)
            ]
        )
        if config.xsa_last_n > 0:
            for idx in range(max(0, config.num_layers - config.xsa_last_n), config.num_layers):
                self.blocks[idx].attn.use_xsa = True
        self.final_norm = RMSNorm()
        self.lm_head = None if config.tie_embeddings else CastedLinear(config.model_dim, runtime.vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        num_layers = len(self.blocks)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if ".proj" in name:
                        with torch.no_grad():
                            module.weight.mul_(1.0 / math.sqrt(2 * num_layers))

    def forward_features(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = self.forward_features(input_ids)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        logits = self.forward_logits(input_ids).reshape(-1, self.vocab_size)
        return F.cross_entropy(logits.float(), target_ids.reshape(-1), reduction="mean")


def collect_calibration_stats(
    model: GPT,
    runtime: RuntimeConfig,
    config: TrainConfig,
    device: torch.device,
) -> dict[str, Tensor]:
    if config.calibration_source == "none" or config.calibration_batches <= 0:
        return {}

    sums: dict[str, Tensor] = {}
    counts: dict[str, int] = {}
    hooks = []

    def make_hook(name: str):
        weight_name = f"{name}.weight"

        def _hook(module, inputs):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
            sq_mean = x.square().mean(dim=0)
            if weight_name not in sums:
                sums[weight_name] = sq_mean
                counts[weight_name] = 1
            else:
                sums[weight_name] += sq_mean
                counts[weight_name] += 1

        return _hook

    for name, module in model.named_modules():
        if isinstance(module, CastedLinear):
            hooks.append(module.register_forward_pre_hook(make_hook(name)))

    was_training = model.training
    model.eval()
    with torch.inference_mode():
        if config.calibration_source == "train_stream":
            loader = DistributedTokenLoader(runtime.train_files, rank=0, world_size=1, device=device)
            batch_tokens = runtime.train_seq_len * max(1, min(8, config.calibration_batches))
            for _ in range(config.calibration_batches):
                x, _ = loader.next_batch(batch_tokens, runtime.train_seq_len, grad_accum_steps=1)
                model.forward_logits(x)
        elif config.calibration_source == "self_generated":
            tokens = torch.zeros((1, runtime.train_seq_len), dtype=torch.int64, device=device)
            for _ in range(config.calibration_batches):
                logits = model.forward_logits(tokens)
                if config.calibration_temperature > 0:
                    probs = torch.softmax(logits[:, -1] / config.calibration_temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
                tokens = torch.cat((tokens[:, 1:], next_token), dim=1)
                model.forward_logits(tokens)
        else:
            raise ValueError(f"Unsupported CALIBRATION_SOURCE={config.calibration_source!r}")

    for hook in hooks:
        hook.remove()
    if was_training:
        model.train()
    return {name: (sums[name] / counts[name]).detach().cpu() for name in sums}


def compile_model(model: nn.Module) -> nn.Module:
    return torch.compile(model, dynamic=False, fullgraph=True)


def build_result_row(
    runtime: RuntimeConfig,
    config: TrainConfig,
    metadata,
    quant_spec,
    config_hash: str,
    status: str,
    artifact_bytes: int | None,
    roundtrip_bpb: float | None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "run_id": metadata.run_id,
        "parent_id": metadata.parent_id,
        "hypothesis_family": metadata.hypothesis_family,
        "hypothesis_statement": metadata.hypothesis_statement,
        "tags": list(metadata.tags),
        "config_hash": config_hash,
        "proxy_bpb": roundtrip_bpb if runtime.is_proxy else None,
        "final_bpb": roundtrip_bpb if not runtime.is_proxy else None,
        "artifact_bytes": artifact_bytes,
        "artifact_cap_bytes": runtime.artifact_cap_bytes,
        "artifact_ok": artifact_bytes is not None and artifact_bytes <= runtime.artifact_cap_bytes,
        "seed": metadata.seed,
        "status": status,
        "search_stage": runtime.stage,
        "search_env": runtime.search_env,
        "quant_mode": quant_spec.mode,
        "compression": quant_spec.compression,
        "train_shards": len(runtime.train_files),
        "val_shards": len(runtime.val_files),
        "error": error,
    }


def _main() -> None:
    runtime = build_runtime_config()
    quant_spec = default_quantization_spec()
    config = build_train_config(runtime)
    metadata = build_experiment_metadata(seed=config.seed)
    config_hash = stable_hash(
        {
            "runtime": summarize_runtime(runtime),
            "train": config.to_record(),
            "quant": {
                "mode": quant_spec.mode,
                "compression": quant_spec.compression,
                "compression_level": quant_spec.compression_level,
                "clip_percentile": quant_spec.clip_percentile,
                "selective_prune_fraction": quant_spec.selective_prune_fraction,
            },
            "tags": metadata.tags,
        }
    )

    if quant_spec.mode == "int6":
        CastedLinear._qat_qmax = 31
    elif quant_spec.mode == "int8":
        CastedLinear._qat_qmax = 127
    else:
        raise ValueError(f"Unsupported quant mode {quant_spec.mode!r}")

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    master_process = rank == 0

    try:
        if world_size <= 0:
            raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
        if 8 % world_size != 0:
            raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
        grad_accum_steps = 8 // world_size
        grad_scale = 1.0 / grad_accum_steps
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        if distributed:
            dist.init_process_group(backend="nccl", device_id=device)
            dist.barrier()

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

        enable_cudnn_sdp(False)
        enable_flash_sdp(True)
        enable_mem_efficient_sdp(False)
        enable_math_sdp(False)

        logfile = None
        if master_process:
            os.makedirs("logs", exist_ok=True)
            logfile = f"logs/{metadata.run_id}.txt"
            print(logfile)

        def log0(msg: str, console: bool = True) -> None:
            if not master_process:
                return
            if console:
                print(msg)
            if logfile is not None:
                with open(logfile, "a", encoding="utf-8") as handle:
                    print(msg, file=handle)

        script_paths = [Path(__file__), Path(__file__).with_name("prepare.py"), Path(__file__).with_name("train_gpt.py")]
        for script_path in script_paths:
            if script_path.exists():
                log0(script_path.read_text(encoding="utf-8"), console=False)
                log0("=" * 100, console=False)
        log0(f"Running Python {sys.version}", console=False)
        log0(f"Running PyTorch {torch.__version__}", console=False)
        log0(
            subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
            console=False,
        )
        log0("=" * 100, console=False)

        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

        sp = load_tokenizer(runtime)
        val_tokens = load_validation_tokens(runtime)
        luts = build_sentencepiece_luts(sp, runtime.vocab_size, device)
        log0(f"search_stage:{runtime.stage} search_env:{runtime.search_env}")
        log0(f"run_id:{metadata.run_id} parent_id:{metadata.parent_id} hypothesis_family:{metadata.hypothesis_family}")
        log0(f"hypothesis_statement:{metadata.hypothesis_statement}")
        log0(f"tags:{','.join(metadata.tags)} config_hash:{config_hash}")
        log0(f"tokenizer_path:{runtime.tokenizer_path}")
        log0(f"train_loader:dataset:{runtime.data_path.name} train_shards:{len(runtime.train_files)}")
        log0(f"val_loader:tokens:{val_tokens.numel() - 1} eval_seq_len:{runtime.eval_seq_len}")

        base_model = GPT(runtime, config).to(device).bfloat16()
        for module in base_model.modules():
            if isinstance(module, CastedLinear):
                module.float()
        restore_low_dim_params_to_fp32(base_model, quant_spec.control_tensor_name_patterns)
        compiled_model = compile_model(base_model)
        model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

        block_named_params = list(base_model.blocks.named_parameters())
        matrix_params = [
            param
            for name, param in block_named_params
            if param.ndim == 2 and not any(pattern in name for pattern in quant_spec.control_tensor_name_patterns)
        ]
        scalar_params = [
            param
            for name, param in block_named_params
            if param.ndim < 2 or any(pattern in name for pattern in quant_spec.control_tensor_name_patterns)
        ]
        if base_model.bigram is not None:
            scalar_params.append(base_model.bigram.scale)
            matrix_params.extend([p for p in [base_model.bigram.proj.weight if base_model.bigram.proj is not None else None] if p is not None])
        token_lr = config.tied_embed_lr if config.tie_embeddings else config.embed_lr
        token_param_groups = [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
        if base_model.bigram is not None:
            token_param_groups.append({"params": [base_model.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
        optimizer_tok = torch.optim.Adam(
            token_param_groups,
            betas=(config.beta1, config.beta2),
            eps=config.adam_eps,
            fused=True,
        )
        optimizer_muon = Muon(
            matrix_params,
            lr=config.matrix_lr,
            momentum=config.muon_momentum,
            backend_steps=config.muon_backend_steps,
        )
        for group in optimizer_muon.param_groups:
            group["base_lr"] = config.matrix_lr
        optimizer_scalar = torch.optim.Adam(
            [{"params": scalar_params, "lr": config.scalar_lr, "base_lr": config.scalar_lr}],
            betas=(config.beta1, config.beta2),
            eps=config.adam_eps,
            fused=True,
        )
        optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
        if base_model.bigram is not None and base_model.bigram.proj is not None:
            base_model.bigram.proj.weight._muon_managed = True
        if base_model.lm_head is not None:
            optimizer_head = torch.optim.Adam(
                [{"params": [base_model.lm_head.weight], "lr": config.head_lr, "base_lr": config.head_lr}],
                betas=(config.beta1, config.beta2),
                eps=config.adam_eps,
                fused=True,
            )
            optimizers.insert(1, optimizer_head)

        n_params = sum(param.numel() for param in base_model.parameters())
        log0(f"model_params:{n_params}")
        log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
        log0(f"attention_mode:gqa num_heads:{config.num_heads} num_kv_heads:{config.num_kv_heads}")
        log0(
            f"baseline:layers={config.num_layers} model_dim={config.model_dim} mlp_mult={config.mlp_mult} "
            f"tie_embeddings={config.tie_embeddings} rope_dims={config.rope_dims} xsa_last_n={config.xsa_last_n}"
        )
        log0(
            f"toggles:bigram={int(config.enable_bigram_hash)} late_qat_threshold={config.late_qat_threshold:.3f} "
            f"openelm_style={int(config.openelm_style)} swiglu={int(config.openelm_use_swiglu)}"
        )
        log0(
            f"train_batch_tokens:{config.train_batch_tokens} train_seq_len:{runtime.train_seq_len} "
            f"iterations:{config.iterations} warmup_steps:{config.warmup_steps} "
            f"max_wallclock_seconds:{runtime.max_wallclock_seconds:.3f}"
        )

        train_loader = DistributedTokenLoader(runtime.train_files, rank, world_size, device)

        def zero_grad_all() -> None:
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)

        max_wallclock_ms = 1000.0 * runtime.max_wallclock_seconds if runtime.max_wallclock_seconds > 0 else None

        def lr_mul(step: int, elapsed_ms: float) -> float:
            if config.warmdown_iters <= 0:
                return 1.0
            if max_wallclock_ms is None:
                warmdown_start = max(config.iterations - config.warmdown_iters, 0)
                if warmdown_start <= step < config.iterations:
                    return max((config.iterations - step) / max(config.warmdown_iters, 1), 0.0)
                return 1.0
            step_ms = elapsed_ms / max(step, 1)
            warmdown_ms = config.warmdown_iters * step_ms
            remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
            return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

        if config.warmup_steps > 0:
            initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
            initial_optimizer_states = [copy.deepcopy(optimizer.state_dict()) for optimizer in optimizers]
            model.train()
            for warmup_step in range(config.warmup_steps):
                zero_grad_all()
                for micro_step in range(grad_accum_steps):
                    if distributed:
                        model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                    x, y = train_loader.next_batch(config.train_batch_tokens, runtime.train_seq_len, grad_accum_steps)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                        warmup_loss = model(x, y)
                    (warmup_loss * grad_scale).backward()
                for optimizer in optimizers:
                    optimizer.step()
                zero_grad_all()
                if config.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == config.warmup_steps:
                    log0(f"warmup_step:{warmup_step + 1}/{config.warmup_steps}")
            base_model.load_state_dict(initial_model_state, strict=True)
            for optimizer, state in zip(optimizers, initial_optimizer_states, strict=True):
                optimizer.load_state_dict(state)
            zero_grad_all()
            if distributed:
                model.require_backward_grad_sync = True
            train_loader = DistributedTokenLoader(runtime.train_files, rank, world_size, device)

        training_time_ms = 0.0
        stop_after_step: int | None = None
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        step = 0
        while True:
            last_step = step == config.iterations or (stop_after_step is not None and step >= stop_after_step)
            should_validate = last_step or (runtime.val_loss_every > 0 and step % runtime.val_loss_every == 0)
            if should_validate:
                torch.cuda.synchronize()
                training_time_ms += 1000.0 * (time.perf_counter() - t0)
                val_loss, val_bpb = eval_val(
                    runtime,
                    model,
                    rank,
                    world_size,
                    device,
                    grad_accum_steps,
                    val_tokens,
                    luts,
                )
                log0(
                    f"step:{step}/{config.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                    f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
                )
                torch.cuda.synchronize()
                t0 = time.perf_counter()

            if last_step:
                if stop_after_step is not None and step < config.iterations:
                    log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{config.iterations}")
                break

            elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
            scale = lr_mul(step, elapsed_ms)
            if config.late_qat_threshold > 0 and scale < config.late_qat_threshold and not CastedLinear._qat_enabled:
                CastedLinear._qat_enabled = True
                log0(f"late_qat:enabled step:{step} scale:{scale:.4f}")
            zero_grad_all()
            train_loss = torch.zeros((), device=device)
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(config.train_batch_tokens, runtime.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    loss = model(x, y)
                train_loss += loss.detach()
                (loss * grad_scale).backward()
            train_loss /= grad_accum_steps

            frac = min(step / config.muon_momentum_warmup_steps, 1.0) if config.muon_momentum_warmup_steps > 0 else 1.0
            muon_momentum = (1.0 - frac) * config.muon_momentum_warmup_start + frac * config.muon_momentum
            for group in optimizer_muon.param_groups:
                group["momentum"] = muon_momentum
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    group["lr"] = group["base_lr"] * scale

            if config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_clip_norm)
            for optimizer in optimizers:
                optimizer.step()
            zero_grad_all()

            step += 1
            approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
            should_log_train = (
                runtime.train_log_every > 0 and (step <= 10 or step % runtime.train_log_every == 0 or stop_after_step is not None)
            )
            if should_log_train:
                log0(
                    f"step:{step}/{config.iterations} train_loss:{train_loss.item():.4f} "
                    f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
                )

            reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
            if distributed and max_wallclock_ms is not None:
                reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
                dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
                reached_cap = bool(reached_cap_tensor.item())
            if stop_after_step is None and reached_cap:
                stop_after_step = step

        log0(
            f"peak memory allocated:{torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
            f"reserved:{torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
        )

        if master_process:
            torch.save(base_model.state_dict(), "final_model.pt")
            model_bytes = Path("final_model.pt").stat().st_size
            code_bytes = count_code_bytes(runtime.artifact_code_paths)
            log0(f"Serialized model fp32/bf16: {model_bytes} bytes")
            log0(f"Code size across autoresearch boundary: {code_bytes} bytes")

        calibration_stats = {}
        if master_process:
            calibration_stats = collect_calibration_stats(base_model, runtime, config, device)
            log0(
                f"calibration:source={config.calibration_source} tensors={len(calibration_stats)} "
                f"batches={config.calibration_batches}"
            )

        if distributed:
            dist.barrier()

        quant_blob = b""
        artifact_total_bytes = 0
        quant_path = Path(f"final_model.{quant_spec.mode}.ptz")
        if master_process:
            quant_payload, quant_stats = quantize_state_dict(base_model.state_dict(), quant_spec, calibration_stats=calibration_stats)
            from prepare import compress_quantized_payload

            quant_blob, raw_bytes = compress_quantized_payload(quant_payload, quant_spec)
            quant_path.write_bytes(quant_blob)
            artifact_total_bytes = artifact_bytes_for_blob(quant_blob, runtime.artifact_code_paths)
            payload_ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["quant_payload_bytes"], 1)
            log0(
                f"Serialized model {quant_spec.mode}+{quant_spec.compression}: {len(quant_blob)} bytes "
                f"(payload:{quant_stats['quant_payload_bytes']} raw_torch:{raw_bytes} payload_ratio:{payload_ratio:.2f}x)"
            )
            log0(f"Total artifact size: {artifact_total_bytes} bytes cap:{runtime.artifact_cap_bytes}")

        if distributed:
            dist.barrier()
        quant_blob_disk = quant_path.read_bytes()
        if not master_process:
            artifact_total_bytes = artifact_bytes_for_blob(quant_blob_disk, runtime.artifact_code_paths)
        quant_state = decompress_quantized_payload(quant_blob_disk, quant_spec)
        dequantized_state = dequantize_state_dict(quant_state)
        CastedLinear._qat_enabled = False

        eval_model = GPT(runtime, config).to(device).bfloat16()
        for module in eval_model.modules():
            if isinstance(module, CastedLinear):
                module.float()
        restore_low_dim_params_to_fp32(eval_model, quant_spec.control_tensor_name_patterns)
        eval_model.load_state_dict(dequantized_state, strict=True)
        compiled_eval = compile_model(eval_model)
        torch.cuda.synchronize()
        t_qeval = time.perf_counter()
        q_val_loss, q_val_bpb = eval_val(
            runtime,
            compiled_eval,
            rank,
            world_size,
            device,
            grad_accum_steps,
            val_tokens,
            luts,
        )
        torch.cuda.synchronize()
        log0(
            f"final_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
            f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
        )
        log0(f"final_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

        if master_process:
            append_result_row(
                runtime.results_path,
                build_result_row(runtime, config, metadata, quant_spec, config_hash, "completed", artifact_total_bytes, q_val_bpb),
            )

    except Exception as exc:
        if master_process:
            append_result_row(
                runtime.results_path,
                build_result_row(runtime, config, metadata, quant_spec, config_hash, "failed", None, None, error=str(exc)),
            )
        raise
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
