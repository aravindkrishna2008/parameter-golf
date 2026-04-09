# Parameter Golf Autoresearch Program

## Objective Order

1. `final_bpb`
2. `artifact_bytes`
3. training stability

Lower is better for `final_bpb` and `artifact_bytes`.

## Grounding

This repository is intentionally grounded in the current winning small-Transformer family and in a quantization-first search strategy.

- Start from an 11-layer decoder-only Transformer at `d_model=512`.
- Use tied embeddings, RoPE, and GQA with 8 query heads and 4 KV heads.
- Treat quantization and calibration as the main optimization axis.
- Do not spend early search cycles on TTT or broader architecture replacement.

## Immutable Boundary

`prepare.py` is immutable and owns:

- tokenizer choice
- dataset slices
- evaluation settings
- compression
- artifact-size accounting
- shared utilities
- result logging

`train.py` is the only intended mutation surface.

## Mutable Surface

`train.py` may change:

- model config
- optimizer config
- quantization hooks
- QAT schedule
- one contest-specific trick at a time

`train.py` may not change:

- tokenizer
- dataset definition
- validation logic
- artifact accounting

Tokenizer work must happen on a dedicated branch explicitly labeled as tokenizer work.

## Search Environments

### Proxy: Google Colab

- `SEARCH_STAGE=proxy`
- fixed proxy shard slice and proxy validation slice from `prepare.py`
- short single-seed experiments
- fixed wall-clock budget
- report `proxy_bpb` and `artifact_bytes` every run

### Authority: DigitalOcean Droplet

- `SEARCH_STAGE=authoritative`
- full tokenizer
- full packaging pipeline
- compressed artifact generation
- fixed multi-seed promotion validation

Do not validate every proxy win. Run one daily truth pass on the top 1-3 proxy candidates.

## Search Ladder

Run this ladder in order:

1. `quant`
Improve calibration and late-QAT scheduling first.

2. `alloc`
Tune parameter allocation under a fixed compressed budget: depth vs FFN width first, then GQA and KV ratio.

3. `embed`
Test embedding-efficiency changes like BigramHash only after the quantization stack is stable.

4. `xsa`, `rope`, `prune`
Test zero- or near-zero-parameter contest tricks one at a time.

5. `openelm-fork`
Test one clean OpenELM-derived fork: pre-norm RMSNorm, bias-free linears, SwiGLU, and non-uniform late-layer widening.

Leave TTT and broader architectural departures out of the main branch until the quantization-first branch plateaus.

## Run Rules

- Every experiment must carry one or more tags from:
  `quant`, `qat`, `alloc`, `embed`, `xsa`, `rope`, `prune`, `openelm-fork`, `optimizer`
- The agent may change only one hypothesis family per run.
- Every run must compare against the current champion.
- Every run must explain the intended mechanism in one sentence.
- Stop exploring a family after three non-improving proposals.
- Merge policy is single-variable wins only.
- Compound patches do not reach the champion branch unless each component already won in isolation.

## Promotion Rules

Promote a proxy candidate only if it:

- improves the proxy champion by at least `0.003` proxy bpb, or
- materially reduces compressed bytes at flat proxy loss

Accept a new authoritative champion only if it:

- fits under the artifact cap
- wins on a fixed multi-seed average
- survives the full packaging pipeline

## Required Logging

Every run must append a row to `results.jsonl` with:

- `run_id`
- `parent_id`
- `hypothesis_family`
- `config_hash`
- `proxy_bpb`
- `final_bpb`
- `artifact_bytes`
- `seed`
- `status`

## Baseline Discipline

Baseline defaults should remain:

- 11 layers
- 512 model dim
- tied embeddings
- RoPE
- 8 query heads / 4 KV heads
- 3x FFN expansion

Keep these toggles off until they win in isolation:

- BigramHash
- XSA
- partial RoPE
- late QAT
- selective pruning
- OpenELM-style non-uniform widening
