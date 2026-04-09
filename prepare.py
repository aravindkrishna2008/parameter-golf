from __future__ import annotations

import glob
import hashlib
import io
import json
import math
import os
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
from torch import Tensor, nn

try:
    import zstandard as zstd
except ImportError:
    zstd = None


REPO_ROOT = Path(__file__).resolve().parent
RESULTS_PATH = REPO_ROOT / "results.jsonl"
ARTIFACT_CAP_BYTES = 16_000_000
DEFAULT_CODE_PATHS = ("prepare.py", "train.py", "train_gpt.py")
ALLOWED_EXPERIMENT_TAGS = {
    "quant",
    "qat",
    "alloc",
    "embed",
    "xsa",
    "rope",
    "prune",
    "openelm-fork",
    "optimizer",
}
CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale",
    "mlp_scale",
    "resid_mix",
    "q_gain",
    "skip_weight",
    "bigram.scale",
    "ve_layer_scale",
    "ve_shared.scale",
)


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or value == "" else float(value)


def _paths_from_glob(pattern: str) -> tuple[Path, ...]:
    return tuple(Path(p) for p in sorted(glob.glob(pattern)))


@dataclass(frozen=True)
class RuntimeConfig:
    stage: str
    search_env: str
    data_path: Path
    train_files: tuple[Path, ...]
    val_files: tuple[Path, ...]
    tokenizer_path: Path
    vocab_size: int
    train_seq_len: int
    eval_seq_len: int
    val_batch_size: int
    val_loss_every: int
    train_log_every: int
    max_wallclock_seconds: float
    artifact_cap_bytes: int
    artifact_code_paths: tuple[Path, ...]
    proxy_train_shards: int
    proxy_val_tokens: int

    @property
    def is_proxy(self) -> bool:
        return self.stage == "proxy"

    @property
    def results_path(self) -> Path:
        return RESULTS_PATH


@dataclass(frozen=True)
class ExperimentMetadata:
    run_id: str
    parent_id: str | None
    hypothesis_family: str
    hypothesis_statement: str
    tags: tuple[str, ...]
    seed: int


@dataclass(frozen=True)
class ValidationLUTs:
    base_bytes: Tensor
    has_leading_space: Tensor
    is_boundary_token: Tensor


@dataclass(frozen=True)
class QuantizationSpec:
    mode: str
    compression: str
    compression_level: int
    clip_percentile: float
    keep_float_max_numel: int
    store_dtype: torch.dtype
    per_row_scale_dtype: torch.dtype
    keep_float_name_patterns: tuple[str, ...]
    control_tensor_name_patterns: tuple[str, ...]
    calibration_multipliers: tuple[float, ...]
    selective_prune_fraction: float


def build_runtime_config() -> RuntimeConfig:
    stage = _env_str("SEARCH_STAGE", "proxy").strip().lower()
    if stage not in {"proxy", "authoritative"}:
        raise ValueError(f"SEARCH_STAGE must be proxy or authoritative, got {stage!r}")
    search_env_default = "colab" if stage == "proxy" else "droplet"
    search_env = _env_str("SEARCH_ENV", search_env_default).strip().lower()
    data_path = Path(_env_str("DATA_PATH", "./data/datasets/fineweb10B_sp1024")).resolve()
    train_pattern = str(data_path / "fineweb_train_*.bin")
    val_pattern = str(data_path / "fineweb_val_*.bin")
    train_files = _paths_from_glob(train_pattern)
    val_files = _paths_from_glob(val_pattern)
    if not train_files:
        raise FileNotFoundError(f"No training files found for {train_pattern}")
    if not val_files:
        raise FileNotFoundError(f"No validation files found for {val_pattern}")

    proxy_train_shards = _env_int("PROXY_TRAIN_SHARDS", 4)
    proxy_val_tokens = _env_int("PROXY_VAL_TOKENS", 2_097_152)
    if stage == "proxy":
        train_files = train_files[:proxy_train_shards]
        if not train_files:
            raise ValueError("Proxy mode selected zero training shards")

    code_paths = tuple(
        (REPO_ROOT / rel.strip()).resolve()
        for rel in _env_str("ARTIFACT_CODE_PATHS", ",".join(DEFAULT_CODE_PATHS)).split(",")
        if rel.strip()
    )
    eval_seq_len = _env_int("EVAL_SEQ_LEN", _env_int("TRAIN_SEQ_LEN", 2048))
    val_batch_size_default = 262_144 if stage == "proxy" else 524_288
    val_loss_every_default = 250 if stage == "proxy" else 1000
    train_log_every_default = 100 if stage == "proxy" else 200
    wallclock_default = 180.0 if stage == "proxy" else 600.0
    return RuntimeConfig(
        stage=stage,
        search_env=search_env,
        data_path=data_path,
        train_files=train_files,
        val_files=val_files,
        tokenizer_path=Path(_env_str("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")).resolve(),
        vocab_size=_env_int("VOCAB_SIZE", 1024),
        train_seq_len=_env_int("TRAIN_SEQ_LEN", 2048),
        eval_seq_len=eval_seq_len,
        val_batch_size=_env_int("VAL_BATCH_SIZE", val_batch_size_default),
        val_loss_every=_env_int("VAL_LOSS_EVERY", val_loss_every_default),
        train_log_every=_env_int("TRAIN_LOG_EVERY", train_log_every_default),
        max_wallclock_seconds=_env_float("MAX_WALLCLOCK_SECONDS", wallclock_default),
        artifact_cap_bytes=_env_int("ARTIFACT_CAP_BYTES", ARTIFACT_CAP_BYTES),
        artifact_code_paths=code_paths,
        proxy_train_shards=proxy_train_shards,
        proxy_val_tokens=proxy_val_tokens,
    )


def build_experiment_metadata(seed: int) -> ExperimentMetadata:
    tags = tuple(tag.strip() for tag in _env_str("EXPERIMENT_TAGS", _env_str("HYPOTHESIS_FAMILY", "quant")).split(",") if tag.strip())
    invalid = [tag for tag in tags if tag not in ALLOWED_EXPERIMENT_TAGS]
    if invalid:
        raise ValueError(f"Unsupported experiment tag(s): {invalid}. Allowed tags: {sorted(ALLOWED_EXPERIMENT_TAGS)}")
    if not tags:
        raise ValueError("At least one experiment tag is required")
    return ExperimentMetadata(
        run_id=_env_str("RUN_ID", "manual"),
        parent_id=os.environ.get("PARENT_ID") or None,
        hypothesis_family=_env_str("HYPOTHESIS_FAMILY", tags[0]),
        hypothesis_statement=_env_str(
            "HYPOTHESIS_STATEMENT",
            "Quantization-first baseline: improve compressed bpb before adding architectural tricks.",
        ),
        tags=tags,
        seed=seed,
    )


def default_quantization_spec() -> QuantizationSpec:
    compression = _env_str("ARTIFACT_COMPRESSION", "zstd" if zstd is not None else "zlib").lower()
    if compression == "zstd" and zstd is None:
        compression = "zlib"
    compression_level = _env_int("ARTIFACT_COMPRESSION_LEVEL", 22 if compression == "zstd" else 9)
    return QuantizationSpec(
        mode=_env_str("EXPORT_QUANT_MODE", "int6").lower(),
        compression=compression,
        compression_level=compression_level,
        clip_percentile=_env_float("EXPORT_CLIP_PERCENTILE", 99.99984),
        keep_float_max_numel=_env_int("KEEP_FLOAT_MAX_NUMEL", 65_536),
        store_dtype=torch.float16,
        per_row_scale_dtype=torch.float16,
        keep_float_name_patterns=tuple(
            pattern.strip()
            for pattern in _env_str(
                "KEEP_FLOAT_NAME_PATTERNS",
                ",".join(CONTROL_TENSOR_NAME_PATTERNS),
            ).split(",")
            if pattern.strip()
        ),
        control_tensor_name_patterns=CONTROL_TENSOR_NAME_PATTERNS,
        calibration_multipliers=tuple(
            float(x.strip())
            for x in _env_str("CALIBRATION_MULTIPLIERS", "0.85,0.925,1.0,1.075,1.15").split(",")
            if x.strip()
        ),
        selective_prune_fraction=_env_float("SELECTIVE_PRUNE_FRACTION", 0.0),
    )


def stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def count_code_bytes(paths: Sequence[Path]) -> int:
    return sum(path.stat().st_size for path in paths if path.exists())


def artifact_bytes_for_blob(blob: bytes, code_paths: Sequence[Path]) -> int:
    return len(blob) + count_code_bytes(code_paths)


def append_result_row(path: Path, row: Mapping[str, Any]) -> None:
    serializable = json.dumps(row, sort_keys=True, default=_json_default)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(serializable)
        handle.write("\n")


def summarize_runtime(runtime: RuntimeConfig) -> dict[str, Any]:
    return {
        "stage": runtime.stage,
        "search_env": runtime.search_env,
        "data_path": str(runtime.data_path),
        "train_files": len(runtime.train_files),
        "val_files": len(runtime.val_files),
        "tokenizer_path": str(runtime.tokenizer_path),
        "vocab_size": runtime.vocab_size,
        "train_seq_len": runtime.train_seq_len,
        "eval_seq_len": runtime.eval_seq_len,
        "val_batch_size": runtime.val_batch_size,
        "max_wallclock_seconds": runtime.max_wallclock_seconds,
        "artifact_cap_bytes": runtime.artifact_cap_bytes,
    }


def load_tokenizer(runtime: RuntimeConfig) -> spm.SentencePieceProcessor:
    if runtime.tokenizer_path.suffix != ".model":
        raise ValueError(f"Expected a SentencePiece .model tokenizer, got {runtime.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=str(runtime.tokenizer_path))
    if int(sp.vocab_size()) != runtime.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={runtime.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    return sp


def build_sentencepiece_luts(sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device) -> ValidationLUTs:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return ValidationLUTs(
        base_bytes=torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        has_leading_space=torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        is_boundary_token=torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def load_validation_tokens(runtime: RuntimeConfig) -> Tensor:
    tokens = torch.cat([load_data_shard(file) for file in runtime.val_files]).contiguous()
    if runtime.is_proxy:
        usable = min(tokens.numel() - 1, runtime.proxy_val_tokens)
        usable = (usable // runtime.eval_seq_len) * runtime.eval_seq_len
        if usable <= 0:
            raise ValueError(f"Proxy validation slice is too short for EVAL_SEQ_LEN={runtime.eval_seq_len}")
        return tokens[: usable + 1]
    usable = ((tokens.numel() - 1) // runtime.eval_seq_len) * runtime.eval_seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for EVAL_SEQ_LEN={runtime.eval_seq_len}")
    return tokens[: usable + 1]


class TokenStream:
    def __init__(self, files: Sequence[Path]):
        if not files:
            raise FileNotFoundError("TokenStream requires at least one shard")
        self.files = tuple(files)
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, files: Sequence[Path], rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(files)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


def eval_val(
    runtime: RuntimeConfig,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    luts: ValidationLUTs,
    eval_seq_len: int | None = None,
) -> tuple[float, float]:
    seq_len = eval_seq_len or runtime.eval_seq_len
    local_batch_tokens = runtime.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={runtime.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, EVAL_SEQ_LEN={seq_len}"
        )
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model_was_training = model.training
    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = luts.base_bytes[tgt_ids].to(dtype=torch.int16)
            token_bytes += (
                luts.has_leading_space[tgt_ids] & ~luts.is_boundary_token[prev_ids]
            ).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    if model_was_training:
        model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


def tensor_nbytes(tensor: Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _keep_float_tensor(name: str, tensor: Tensor, spec: QuantizationSpec, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in spec.keep_float_name_patterns):
        return tensor.float().contiguous()
    if tensor.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(tensor.dtype).removeprefix("torch.")
        return tensor.to(dtype=spec.store_dtype).contiguous()
    return tensor


def _quantize_1d_or_scalar(tensor: Tensor, qmax: int, clip_q: float) -> tuple[Tensor, Tensor]:
    t32 = tensor.float()
    clip_abs = float(torch.quantile(t32.abs().flatten(), clip_q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / qmax if clip_abs > 0 else 1.0, dtype=torch.float32)
    quantized = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -qmax, qmax).to(torch.int8)
    return quantized.contiguous(), scale


def _apply_selective_pruning(quantized: Tensor, fraction: float) -> Tensor:
    if fraction <= 0.0 or quantized.numel() == 0:
        return quantized
    k = int(quantized.numel() * fraction)
    if k <= 0:
        return quantized
    flat_abs = quantized.abs().reshape(-1)
    threshold = torch.kthvalue(flat_abs, min(k, flat_abs.numel())).values
    pruned = quantized.clone()
    pruned[quantized.abs() <= threshold] = 0
    return pruned


def _quantize_matrix(
    tensor: Tensor,
    qmax: int,
    clip_q: float,
    spec: QuantizationSpec,
    input_sq_mean: Tensor | None,
) -> tuple[Tensor, Tensor]:
    t32 = tensor.float()
    base_clip = torch.quantile(t32.abs(), clip_q, dim=1) if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32)
    if input_sq_mean is None or input_sq_mean.numel() != t32.shape[1]:
        scale = (base_clip / qmax).clamp_min(1.0 / qmax)
        clipped = torch.maximum(torch.minimum(t32, scale[:, None] * qmax), -scale[:, None] * qmax)
        quantized = torch.clamp(torch.round(clipped / scale[:, None]), -qmax, qmax).to(torch.int8)
        return _apply_selective_pruning(quantized.contiguous(), spec.selective_prune_fraction), scale.to(dtype=spec.per_row_scale_dtype)

    weights = input_sq_mean.to(dtype=torch.float32, device=t32.device).reshape(1, -1)
    best_quantized: Tensor | None = None
    best_scale: Tensor | None = None
    best_error: float | None = None
    for mult in spec.calibration_multipliers:
        scale = ((base_clip * mult) / qmax).clamp_min(1.0 / qmax)
        clipped = torch.maximum(torch.minimum(t32, scale[:, None] * qmax), -scale[:, None] * qmax)
        quantized = torch.clamp(torch.round(clipped / scale[:, None]), -qmax, qmax)
        dequantized = quantized * scale[:, None]
        error = float((((t32 - dequantized) ** 2) * weights).mean().item())
        if best_error is None or error < best_error:
            best_error = error
            best_quantized = quantized.to(torch.int8).contiguous()
            best_scale = scale.to(dtype=spec.per_row_scale_dtype).contiguous()
    if best_quantized is None or best_scale is None:
        raise RuntimeError("Calibration search failed to produce a quantized tensor")
    return _apply_selective_pruning(best_quantized, spec.selective_prune_fraction), best_scale


def quantize_state_dict(
    state_dict: Mapping[str, Tensor],
    spec: QuantizationSpec,
    calibration_stats: Mapping[str, Tensor] | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    if spec.mode not in {"int8", "int6"}:
        raise ValueError(f"Unsupported export quantization mode {spec.mode!r}")
    qmax = 31 if spec.mode == "int6" else 127
    clip_q = spec.clip_percentile / 100.0
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, Any]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "quant_payload_bytes"),
        0,
    )

    for name, tensor in state_dict.items():
        cpu_tensor = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(cpu_tensor.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(cpu_tensor)
        if not cpu_tensor.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = cpu_tensor
            stats["quant_payload_bytes"] += tensor_nbytes(cpu_tensor)
            continue
        if cpu_tensor.numel() <= spec.keep_float_max_numel:
            kept = _keep_float_tensor(name, cpu_tensor, spec, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["quant_payload_bytes"] += tensor_nbytes(kept)
            continue
        stats["num_float_tensors"] += 1
        calibration = calibration_stats.get(name) if calibration_stats is not None else None
        if cpu_tensor.ndim == 2:
            q, s = _quantize_matrix(cpu_tensor, qmax, clip_q, spec, calibration)
            qmeta[name] = {"scheme": "per_row", "axis": 0, "qmax": qmax}
        else:
            q, s = _quantize_1d_or_scalar(cpu_tensor, qmax, clip_q)
            qmeta[name] = {"scheme": "per_tensor", "qmax": qmax}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(cpu_tensor.dtype).removeprefix("torch.")
        stats["quant_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    payload: dict[str, Any] = {
        "__quant_format__": f"{spec.mode}_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
        "qmeta": qmeta,
    }
    if passthrough_orig_dtypes:
        payload["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return payload, stats


def dequantize_state_dict(payload: Mapping[str, Any]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = payload.get("qmeta", {})
    passthrough_orig_dtypes = payload.get("passthrough_orig_dtypes", {})
    for name, quantized in payload["quantized"].items():
        dtype = getattr(torch, payload["dtypes"][name])
        scale = payload["scales"][name]
        meta = qmeta.get(name, {})
        qmax = int(meta.get("qmax", 127))
        if meta.get("scheme") == "per_row" or scale.ndim > 0:
            scale32 = scale.to(dtype=torch.float32)
            out[name] = (quantized.float() * scale32.view(quantized.shape[0], *([1] * (quantized.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            out[name] = (quantized.float() * float(scale.item())).to(dtype=dtype).contiguous()
        if qmax not in {31, 127}:
            raise ValueError(f"Unexpected qmax {qmax} while dequantizing {name}")
    for name, tensor in payload["passthrough"].items():
        restored = tensor.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            restored = restored.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = restored
    return out


def compress_quantized_payload(payload: Mapping[str, Any], spec: QuantizationSpec) -> tuple[bytes, int]:
    raw_buffer = io.BytesIO()
    torch.save(payload, raw_buffer)
    raw_bytes = raw_buffer.getvalue()
    if spec.compression == "zstd":
        if zstd is None:
            raise RuntimeError("zstd compression requested but zstandard is not installed")
        blob = zstd.ZstdCompressor(level=spec.compression_level).compress(raw_bytes)
    elif spec.compression == "zlib":
        blob = zlib.compress(raw_bytes, level=spec.compression_level)
    else:
        raise ValueError(f"Unsupported compression {spec.compression!r}")
    return blob, len(raw_bytes)


def decompress_quantized_payload(blob: bytes, spec: QuantizationSpec) -> Mapping[str, Any]:
    if spec.compression == "zstd":
        if zstd is None:
            raise RuntimeError("zstd compression requested but zstandard is not installed")
        raw = zstd.ZstdDecompressor().decompress(blob)
    elif spec.compression == "zlib":
        raw = zlib.decompress(blob)
    else:
        raise ValueError(f"Unsupported compression {spec.compression!r}")
    return torch.load(io.BytesIO(raw), map_location="cpu")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, (set, tuple)):
        return list(value)
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
