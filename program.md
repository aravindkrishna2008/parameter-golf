# Parameter Golf Autoresearch Program

This repository uses an upstream-style autoresearch loop adapted to the Parameter Golf evaluation rules.

## Immutable Boundary

Read these files before changing anything:

- `README.md` for contest context and launch commands.
- `prepare.py` for the immutable runtime and evaluation harness.
- `train.py` for the mutable model and training loop.
- `autoresearch.py` for the immutable experiment driver.

The boundary is strict:

- Do not modify `prepare.py`.
- Do not modify `autoresearch.py`.
- Do not modify tokenizer choice, dataset slices, validation logic, compression, or artifact accounting from `train.py`.
- Only modify `train.py` during the main search loop.

Tokenizer work is allowed only on a clearly separate tokenizer branch, not in the main autoresearch branch.

## Objective Order

1. Minimize `final_bpb`.
2. Stay under the 16,000,000-byte artifact cap.
3. Prefer simpler winning changes over equally good but more complex ones.

Lower is better for `proxy_bpb`, `final_bpb`, and `artifact_bytes`.

## Setup

Start each fresh autoresearch run with:

```bash
python3 autoresearch.py setup --tag <tag>
```

If you want the driver to create the branch for you, use:

```bash
python3 autoresearch.py setup --tag <tag> --create-branch
```

The expected branch name is `autoresearch/<tag>`.

The setup is not ready until all of the following are true:

- `results.jsonl` exists.
- the cached FineWeb shards exist under `data/datasets/fineweb10B_sp1024/`
- the tokenizer exists at `data/tokenizers/fineweb_1024_bpe.model`

If data is missing, fix the environment first. Do not start the loop until setup passes.

## Main Loop

The first run is always the unmodified baseline on the current `train.py`.

Run one proxy experiment at a time with:

```bash
python3 autoresearch.py experiment \
  --hypothesis-family quant \
  --hypothesis-statement "Late QAT should improve compression without hurting proxy loss."
```

The driver is the source of truth for proxy decisions. It will:

- run the current `train.py` in `SEARCH_STAGE=proxy`
- write logs to `logs/<run_id>.driver.log`
- read the appended `results.jsonl` row
- compare the run against the current proxy champion
- commit `train.py` if the run is a keep
- restore `train.py` to `HEAD` if the run is a discard

Do not hand-roll keep/discard logic outside the driver unless the driver explicitly says git actions were skipped.

## Allowed Changes

Within `train.py`, you may change:

- model shape and allocation
- optimizer settings
- QAT and calibration strategy
- one contest-specific trick at a time

Within the main branch, keep the search grounded in the current strong small-Transformer family:

- baseline around 11 layers and `d_model=512`
- tied embeddings
- RoPE
- 8 query heads / 4 KV heads
- 3x FFN expansion

Prioritize search families in this order:

1. `quant`
2. `alloc`
3. `embed`
4. `xsa`, `rope`, `prune`
5. `openelm-fork`

Do not jump to TTT or broad architecture replacement until the quantization-first path clearly plateaus.

## Run Rules

- Every run needs a one-sentence mechanism via `--hypothesis-statement`.
- Use one hypothesis family per run.
- Prefer small, isolatable diffs.
- Stop pushing a family after roughly three non-improving attempts.
- A flat result with meaningfully smaller artifact bytes can still be a win.
- If a result is basically tied, prefer the simpler code.

## Promotion

Proxy wins are cheap filters. They are not authoritative.

Promote a candidate with the fixed seed set using:

```bash
python3 autoresearch.py promote \
  --hypothesis-family quant \
  --hypothesis-statement "Late QAT should improve compression without hurting final loss."
```

The promotion path is fixed:

- `SEARCH_STAGE=authoritative`
- seeds `42 1337 2025` by default
- one `results.jsonl` row per seed run
- one aggregate `promotion_summary` row appended after all seeds finish

An authoritative champion must:

- complete all promotion seeds
- stay under the artifact cap on every seed
- win on the fixed-seed average
- survive the full packaging path

## Logging And Status

`results.jsonl` is the durable experiment ledger. Do not manually rewrite it.

Useful commands:

```bash
python3 autoresearch.py status
```

This prints the current best proxy experiment and best authoritative promotion.

## Failure Handling

If a run crashes, read `logs/<run_id>.driver.log` and fix the failure in `train.py`.

- If the crash is a simple bug, fix it and rerun.
- If the idea is fundamentally bad, move on.
- Do not edit the immutable runtime just to rescue a weak hypothesis.

## Simplicity Criterion

The goal is not to accumulate hacks. The goal is to find robust wins within the contest rules.

- Keep a small gain if it is clean and understandable.
- Reject a tiny gain if it adds disproportionate complexity.
- Strongly prefer deletions, simplifications, or cleaner formulations when they preserve quality.
