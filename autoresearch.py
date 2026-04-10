from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent
RESULTS_PATH = REPO_ROOT / "results.jsonl"
LOG_DIR = REPO_ROOT / "logs"
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "datasets" / "fineweb10B_sp1024"
DEFAULT_TOKENIZER_PATH = REPO_ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model"
DEFAULT_PROXY_IMPROVEMENT = 0.003
DEFAULT_PROXY_FLAT_LOSS_TOL = 0.0005
DEFAULT_PROXY_BYTE_SAVINGS = 16_384
DEFAULT_AUTHORITATIVE_SEEDS = (42, 1337, 2025)
DEFAULT_AUTHORITATIVE_IMPROVEMENT = 0.0
ALLOWED_MUTATION_FILES = {"train.py"}
IGNORED_WORKTREE_FILES = {"results.jsonl"}


class AutoresearchError(RuntimeError):
    pass


@dataclass(frozen=True)
class Decision:
    value: str
    reason: str
    champion_run_id: str | None = None


def utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def local_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rows.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise AutoresearchError(f"Could not parse {path}:{lineno}: {exc}") from exc
    return rows


def append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True))
        handle.write("\n")


def git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=check,
        text=True,
        capture_output=True,
    )


def git_available() -> bool:
    try:
        git(["rev-parse", "--git-dir"])
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return True


def current_branch() -> str | None:
    if not git_available():
        return None
    try:
        result = git(["branch", "--show-current"])
    except subprocess.CalledProcessError:
        return None
    return result.stdout.strip() or None


def branch_exists(name: str) -> bool:
    try:
        git(["rev-parse", "--verify", f"refs/heads/{name}"])
    except subprocess.CalledProcessError:
        return False
    return True


def tracked_status_paths() -> set[str]:
    if not git_available():
        return set()
    result = git(["status", "--porcelain=v1"])
    paths: set[str] = set()
    for line in result.stdout.splitlines():
        if not line:
            continue
        status = line[:2]
        path = line[3:]
        if "->" in path:
            path = path.split("->", 1)[1].strip()
        if status == "??":
            continue
        paths.add(path)
    return paths


def ensure_results_file() -> None:
    RESULTS_PATH.touch(exist_ok=True)


def default_run_id(stage: str, family: str) -> str:
    safe_family = family.replace("_", "-")
    return f"{local_stamp()}-{stage}-{safe_family}"


def require_run_id_unused(run_id: str, rows: Iterable[dict[str, Any]]) -> None:
    for row in rows:
        if row.get("run_id") == run_id:
            raise AutoresearchError(f"run_id {run_id!r} already exists in {RESULTS_PATH.name}")


def rows_for_run_id(rows: Iterable[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("run_id") == run_id]


def metric_key_for_stage(stage: str) -> str:
    if stage == "proxy":
        return "proxy_bpb"
    if stage == "authoritative":
        return "final_bpb"
    raise AutoresearchError(f"Unsupported stage {stage!r}")


def experiment_rows(rows: Iterable[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    metric_key = metric_key_for_stage(stage)
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if row.get("record_kind", "experiment") != "experiment":
            continue
        if row.get("search_stage") != stage:
            continue
        if row.get("status") != "completed":
            continue
        if not row.get("artifact_ok", False):
            continue
        metric = row.get(metric_key)
        if metric is None:
            continue
        filtered.append(row)
    return filtered


def promotion_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if row.get("record_kind") != "promotion_summary":
            continue
        if row.get("status") != "completed":
            continue
        if not row.get("artifact_ok", False):
            continue
        if row.get("final_bpb_mean") is None:
            continue
        filtered.append(row)
    return filtered


def best_row(rows: Iterable[dict[str, Any]], metric_key: str) -> dict[str, Any] | None:
    usable = [row for row in rows if row.get(metric_key) is not None]
    if not usable:
        return None
    return min(
        usable,
        key=lambda row: (
            float(row[metric_key]),
            int(row.get("artifact_bytes_max", row.get("artifact_bytes", 1 << 60))),
        ),
    )


def proxy_decision(
    candidate: dict[str, Any],
    prior_rows: list[dict[str, Any]],
    *,
    min_improvement: float,
    flat_loss_tol: float,
    min_byte_savings: int,
) -> Decision:
    if candidate.get("status") != "completed":
        return Decision("discard", f"run failed with status={candidate.get('status')}")
    if not candidate.get("artifact_ok", False):
        return Decision("discard", "artifact exceeded cap")
    champion = best_row(experiment_rows(prior_rows, "proxy"), "proxy_bpb")
    if champion is None:
        return Decision("baseline", "first completed proxy run under artifact cap")
    candidate_bpb = float(candidate["proxy_bpb"])
    champion_bpb = float(champion["proxy_bpb"])
    if candidate_bpb <= champion_bpb - min_improvement:
        return Decision(
            "keep",
            f"proxy_bpb improved from {champion_bpb:.6f} to {candidate_bpb:.6f}",
            champion_run_id=str(champion.get("run_id")),
        )
    candidate_bytes = int(candidate.get("artifact_bytes") or 0)
    champion_bytes = int(champion.get("artifact_bytes") or 0)
    if abs(candidate_bpb - champion_bpb) <= flat_loss_tol and candidate_bytes <= champion_bytes - min_byte_savings:
        return Decision(
            "keep",
            "proxy_bpb stayed effectively flat while artifact bytes decreased",
            champion_run_id=str(champion.get("run_id")),
        )
    return Decision(
        "discard",
        f"did not beat champion {champion.get('run_id')} at proxy stage",
        champion_run_id=str(champion.get("run_id")),
    )


def authoritative_decision(
    summary_row: dict[str, Any],
    prior_rows: list[dict[str, Any]],
    *,
    min_improvement: float,
) -> Decision:
    if summary_row.get("status") != "completed":
        return Decision("discard", "promotion did not complete on all seeds")
    if not summary_row.get("artifact_ok", False):
        return Decision("discard", "one or more authoritative seed runs exceeded the artifact cap")
    champion = best_row(promotion_rows(prior_rows), "final_bpb_mean")
    if champion is None:
        return Decision("baseline", "first completed authoritative promotion")
    candidate_mean = float(summary_row["final_bpb_mean"])
    champion_mean = float(champion["final_bpb_mean"])
    if candidate_mean <= champion_mean - min_improvement:
        return Decision(
            "keep",
            f"authoritative mean improved from {champion_mean:.6f} to {candidate_mean:.6f}",
            champion_run_id=str(champion.get("run_id")),
        )
    return Decision(
        "discard",
        f"did not beat authoritative champion {champion.get('run_id')}",
        champion_run_id=str(champion.get("run_id")),
    )


def build_train_command(raw_command: str | None) -> list[str]:
    if raw_command:
        return shlex.split(raw_command)
    return [sys.executable, "train_gpt.py"]


def run_training(run_id: str, env_overrides: dict[str, str], train_command: list[str]) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{run_id}.driver.log"
    env = os.environ.copy()
    env.update(env_overrides)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("# autoresearch driver\n")
        handle.write(f"# run_id={run_id}\n")
        handle.write(f"# command={' '.join(train_command)}\n")
        handle.write(f"# started_at={utc_now()}\n")
        handle.flush()
        proc = subprocess.run(
            train_command,
            cwd=REPO_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        handle.write(f"# finished_at={utc_now()}\n")
        handle.write(f"# returncode={proc.returncode}\n")
    return proc.returncode


def safe_for_git_actions() -> tuple[bool, str]:
    if not git_available():
        return False, "git not available"
    tracked = tracked_status_paths() - IGNORED_WORKTREE_FILES
    unexpected = tracked - ALLOWED_MUTATION_FILES
    if unexpected:
        return False, f"tracked changes outside train.py: {', '.join(sorted(unexpected))}"
    return True, ""


def clean_for_setup() -> tuple[bool, str]:
    if not git_available():
        return False, "git not available"
    tracked = tracked_status_paths() - IGNORED_WORKTREE_FILES
    if tracked:
        return False, f"tracked changes present: {', '.join(sorted(tracked))}"
    return True, ""


def has_train_py_changes() -> bool:
    return "train.py" in tracked_status_paths()


def maybe_commit_train_py(run_id: str, family: str, statement: str) -> str | None:
    if not has_train_py_changes():
        return None
    message = f"autoresearch: {run_id} [{family}]"
    detail = statement.strip().splitlines()[0]
    if detail:
        message = f"{message}\n\n{detail}"
    git(["add", "--", "train.py"])
    git(["commit", "-m", message])
    result = git(["rev-parse", "--short", "HEAD"])
    return result.stdout.strip() or None


def restore_train_py() -> None:
    git(["restore", "--source=HEAD", "--staged", "--worktree", "--", "train.py"])


def summarize_experiment(row: dict[str, Any]) -> str:
    stage = str(row.get("search_stage"))
    metric_key = metric_key_for_stage(stage)
    metric = row.get(metric_key)
    artifact_bytes = row.get("artifact_bytes")
    return (
        f"run_id={row.get('run_id')} stage={stage} status={row.get('status')} "
        f"{metric_key}={metric} artifact_bytes={artifact_bytes}"
    )


def check_data_ready(data_path: Path, tokenizer_path: Path) -> list[str]:
    issues: list[str] = []
    train_shards = sorted(data_path.glob("fineweb_train_*.bin"))
    val_shards = sorted(data_path.glob("fineweb_val_*.bin"))
    if not train_shards:
        issues.append(f"missing training shards under {data_path}")
    if not val_shards:
        issues.append(f"missing validation shards under {data_path}")
    if not tokenizer_path.exists():
        issues.append(f"missing tokenizer at {tokenizer_path}")
    return issues


def command_setup(args: argparse.Namespace) -> int:
    ensure_results_file()
    data_path = Path(args.data_path).resolve()
    tokenizer_path = Path(args.tokenizer_path).resolve()
    issues = check_data_ready(data_path, tokenizer_path)
    branch = f"autoresearch/{args.tag}" if args.tag else None

    print(f"results_path: {RESULTS_PATH}")
    print(f"current_branch: {current_branch() or 'unknown'}")
    print(f"data_path: {data_path}")
    print(f"tokenizer_path: {tokenizer_path}")
    if issues:
        print("data_status: missing")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("data_status: ready")

    if args.create_branch:
        if not git_available():
            raise AutoresearchError("git is required to create an autoresearch branch")
        safe, reason = clean_for_setup()
        if not safe:
            raise AutoresearchError(f"Refusing to create branch: {reason}")
        if not branch:
            raise AutoresearchError("--create-branch requires --tag")
        if branch_exists(branch):
            raise AutoresearchError(f"Branch {branch!r} already exists")
        git(["checkout", "-b", branch])
        print(f"created_branch: {branch}")
    elif branch:
        print(f"recommended_branch: {branch}")

    if issues:
        print("setup_status: incomplete")
        return 1
    print("setup_status: ready")
    return 0


def command_experiment(args: argparse.Namespace) -> int:
    ensure_results_file()
    rows_before = read_jsonl(RESULTS_PATH)
    run_id = args.run_id or default_run_id("proxy", args.hypothesis_family)
    require_run_id_unused(run_id, rows_before)

    safe_git, git_reason = safe_for_git_actions()
    env_overrides = {
        "RUN_ID": run_id,
        "SEARCH_STAGE": "proxy",
        "SEED": str(args.seed),
        "HYPOTHESIS_FAMILY": args.hypothesis_family,
        "HYPOTHESIS_STATEMENT": args.hypothesis_statement,
        "EXPERIMENT_TAGS": args.tags or args.hypothesis_family,
    }
    if args.parent_id:
        env_overrides["PARENT_ID"] = args.parent_id

    returncode = run_training(run_id, env_overrides, build_train_command(args.train_command))
    rows_after = read_jsonl(RESULTS_PATH)
    matches = rows_for_run_id(rows_after, run_id)
    if not matches:
        raise AutoresearchError(
            f"No results row found for run_id={run_id!r}. See {LOG_DIR / f'{run_id}.driver.log'}"
        )
    candidate = matches[-1]
    decision = proxy_decision(
        candidate,
        rows_before,
        min_improvement=args.min_improvement,
        flat_loss_tol=args.flat_loss_tol,
        min_byte_savings=args.min_byte_savings,
    )

    commit_sha = None
    if decision.value in {"baseline", "keep"} and safe_git:
        commit_sha = maybe_commit_train_py(run_id, args.hypothesis_family, args.hypothesis_statement)
    elif decision.value == "discard" and safe_git and has_train_py_changes():
        restore_train_py()

    print(summarize_experiment(candidate))
    print(f"driver_returncode={returncode}")
    print(f"decision={decision.value}")
    print(f"reason={decision.reason}")
    if decision.champion_run_id:
        print(f"champion_run_id={decision.champion_run_id}")
    if commit_sha:
        print(f"commit={commit_sha}")
    if not safe_git:
        print(f"git_actions=skipped ({git_reason})")
    elif decision.value == "discard":
        print("git_actions=restored train.py to HEAD")
    elif commit_sha:
        print("git_actions=committed train.py")
    else:
        print("git_actions=no-op")
    return 0 if candidate.get("status") == "completed" else max(returncode, 1)


def promotion_summary_row(
    *,
    base_run_id: str,
    args: argparse.Namespace,
    seed_rows: list[dict[str, Any]],
    decision: Decision,
) -> dict[str, Any]:
    completed_rows = [row for row in seed_rows if row.get("status") == "completed"]
    artifact_ok = bool(seed_rows) and all(bool(row.get("artifact_ok", False)) for row in completed_rows) and len(completed_rows) == len(seed_rows)
    final_bpbs = [float(row["final_bpb"]) for row in completed_rows if row.get("final_bpb") is not None]
    artifact_bytes = [int(row["artifact_bytes"]) for row in completed_rows if row.get("artifact_bytes") is not None]
    return {
        "record_kind": "promotion_summary",
        "run_id": base_run_id,
        "parent_id": args.parent_id,
        "hypothesis_family": args.hypothesis_family,
        "hypothesis_statement": args.hypothesis_statement,
        "tags": [tag.strip() for tag in (args.tags or args.hypothesis_family).split(",") if tag.strip()],
        "status": "completed" if len(completed_rows) == len(seed_rows) else "failed",
        "search_stage": "authoritative",
        "seed_count": len(seed_rows),
        "completed_seed_count": len(completed_rows),
        "seed_run_ids": [str(row.get("run_id")) for row in seed_rows],
        "seed_metrics": [
            {
                "run_id": row.get("run_id"),
                "seed": row.get("seed"),
                "status": row.get("status"),
                "final_bpb": row.get("final_bpb"),
                "artifact_bytes": row.get("artifact_bytes"),
                "artifact_ok": row.get("artifact_ok"),
            }
            for row in seed_rows
        ],
        "final_bpb_mean": statistics.fmean(final_bpbs) if final_bpbs else None,
        "final_bpb_std": statistics.pstdev(final_bpbs) if len(final_bpbs) > 1 else 0.0 if final_bpbs else None,
        "artifact_bytes_max": max(artifact_bytes) if artifact_bytes else None,
        "artifact_bytes_mean": statistics.fmean(artifact_bytes) if artifact_bytes else None,
        "artifact_ok": artifact_ok,
        "decision": decision.value,
        "decision_reason": decision.reason,
        "champion_run_id": decision.champion_run_id,
        "created_at": utc_now(),
    }


def command_promote(args: argparse.Namespace) -> int:
    ensure_results_file()
    rows_before = read_jsonl(RESULTS_PATH)
    base_run_id = args.base_run_id or default_run_id("authoritative", args.hypothesis_family)
    require_run_id_unused(base_run_id, rows_before)
    seed_rows: list[dict[str, Any]] = []
    overall_returncode = 0
    for seed in args.seeds:
        run_id = f"{base_run_id}-s{seed}"
        require_run_id_unused(run_id, rows_before)
        env_overrides = {
            "RUN_ID": run_id,
            "SEARCH_STAGE": "authoritative",
            "SEED": str(seed),
            "HYPOTHESIS_FAMILY": args.hypothesis_family,
            "HYPOTHESIS_STATEMENT": args.hypothesis_statement,
            "EXPERIMENT_TAGS": args.tags or args.hypothesis_family,
        }
        if args.parent_id:
            env_overrides["PARENT_ID"] = args.parent_id
        returncode = run_training(run_id, env_overrides, build_train_command(args.train_command))
        overall_returncode = max(overall_returncode, returncode)
        rows_after_seed = read_jsonl(RESULTS_PATH)
        matches = rows_for_run_id(rows_after_seed, run_id)
        if not matches:
            raise AutoresearchError(
                f"No results row found for run_id={run_id!r}. See {LOG_DIR / f'{run_id}.driver.log'}"
            )
        seed_rows.append(matches[-1])
        print(summarize_experiment(matches[-1]))

    summary_stub = {
        "record_kind": "promotion_summary",
        "status": "completed" if all(row.get("status") == "completed" for row in seed_rows) else "failed",
        "artifact_ok": all(bool(row.get("artifact_ok", False)) for row in seed_rows),
        "final_bpb_mean": (
            statistics.fmean(float(row["final_bpb"]) for row in seed_rows if row.get("final_bpb") is not None)
            if all(row.get("final_bpb") is not None for row in seed_rows)
            else None
        ),
    }
    decision = authoritative_decision(summary_stub, rows_before, min_improvement=args.min_improvement)
    summary_row = promotion_summary_row(base_run_id=base_run_id, args=args, seed_rows=seed_rows, decision=decision)
    append_jsonl_row(RESULTS_PATH, summary_row)

    print(f"promotion_run_id={base_run_id}")
    print(f"seed_runs={','.join(summary_row['seed_run_ids'])}")
    print(f"decision={decision.value}")
    print(f"reason={decision.reason}")
    if summary_row.get("final_bpb_mean") is not None:
        print(f"final_bpb_mean={summary_row['final_bpb_mean']:.6f}")
        print(f"final_bpb_std={summary_row['final_bpb_std']:.6f}")
    print(f"artifact_ok={summary_row['artifact_ok']}")
    return 0 if summary_row["status"] == "completed" else max(overall_returncode, 1)


def command_status(_: argparse.Namespace) -> int:
    rows = read_jsonl(RESULTS_PATH)
    proxy = best_row(experiment_rows(rows, "proxy"), "proxy_bpb")
    promo = best_row(promotion_rows(rows), "final_bpb_mean")
    print(f"results_path: {RESULTS_PATH}")
    if proxy:
        print(
            "best_proxy: "
            f"run_id={proxy.get('run_id')} proxy_bpb={float(proxy['proxy_bpb']):.6f} "
            f"artifact_bytes={proxy.get('artifact_bytes')}"
        )
    else:
        print("best_proxy: none")
    if promo:
        print(
            "best_authoritative: "
            f"run_id={promo.get('run_id')} final_bpb_mean={float(promo['final_bpb_mean']):.6f} "
            f"artifact_bytes_max={promo.get('artifact_bytes_max')}"
        )
    else:
        print("best_authoritative: none")
    print(f"total_rows: {len(rows)}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Immutable experiment driver for Parameter Golf autoresearch.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup = subparsers.add_parser("setup", help="Validate the repo state and optionally create an autoresearch branch.")
    setup.add_argument("--tag", help="Run tag for the autoresearch/<tag> branch.")
    setup.add_argument("--create-branch", action="store_true", help="Create autoresearch/<tag> after validation.")
    setup.add_argument("--data-path", default=str(DEFAULT_DATA_PATH), help="Dataset directory to validate.")
    setup.add_argument("--tokenizer-path", default=str(DEFAULT_TOKENIZER_PATH), help="Tokenizer path to validate.")
    setup.set_defaults(func=command_setup)

    experiment = subparsers.add_parser("experiment", help="Run one proxy experiment and make a keep/discard decision.")
    experiment.add_argument("--run-id", help="Unique run identifier. Defaults to a timestamped proxy run id.")
    experiment.add_argument("--hypothesis-family", default="quant", help="Primary hypothesis family tag.")
    experiment.add_argument("--tags", help="Comma-separated tags passed as EXPERIMENT_TAGS.")
    experiment.add_argument("--hypothesis-statement", required=True, help="One-sentence description of the mechanism under test.")
    experiment.add_argument("--seed", type=int, default=1337, help="Seed for the proxy run.")
    experiment.add_argument("--parent-id", help="Optional parent run id.")
    experiment.add_argument("--train-command", help="Override the training launcher. Default: python train_gpt.py")
    experiment.add_argument("--min-improvement", type=float, default=DEFAULT_PROXY_IMPROVEMENT, help="Minimum proxy_bpb improvement required to keep a win.")
    experiment.add_argument("--flat-loss-tol", type=float, default=DEFAULT_PROXY_FLAT_LOSS_TOL, help="Loss tolerance for artifact-size tie-breaks.")
    experiment.add_argument("--min-byte-savings", type=int, default=DEFAULT_PROXY_BYTE_SAVINGS, help="Required artifact-byte reduction when proxy_bpb is effectively flat.")
    experiment.set_defaults(func=command_experiment)

    promote = subparsers.add_parser("promote", help="Run fixed-seed authoritative promotion and append an aggregate summary row.")
    promote.add_argument("--base-run-id", help="Unique run identifier for the promotion summary.")
    promote.add_argument("--hypothesis-family", default="quant", help="Primary hypothesis family tag.")
    promote.add_argument("--tags", help="Comma-separated tags passed as EXPERIMENT_TAGS.")
    promote.add_argument("--hypothesis-statement", required=True, help="One-sentence description of the mechanism under test.")
    promote.add_argument("--parent-id", help="Optional parent run id.")
    promote.add_argument("--train-command", help="Override the training launcher. Default: python train_gpt.py")
    promote.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_AUTHORITATIVE_SEEDS),
        help="Fixed seed set for authoritative promotion.",
    )
    promote.add_argument(
        "--min-improvement",
        type=float,
        default=DEFAULT_AUTHORITATIVE_IMPROVEMENT,
        help="Minimum authoritative mean improvement required to keep a promotion win.",
    )
    promote.set_defaults(func=command_promote)

    status = subparsers.add_parser("status", help="Show the current proxy and authoritative champions.")
    status.set_defaults(func=command_status)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return int(args.func(args))
    except AutoresearchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
