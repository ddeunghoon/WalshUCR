from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Haar instance consistency across GPU VQSD runs.")
    parser.add_argument(
        "result_dirs",
        nargs="*",
        help="Optional result roots or instance/model output dirs. The script scans for summaries/*_summary.json.",
    )
    parser.add_argument("--n-sys", type=int, default=4)
    parser.add_argument("--M", type=int, default=20)
    parser.add_argument("--instance-start", type=int, default=0)
    parser.add_argument("--instance-end", type=int, default=9)
    parser.add_argument("--models", nargs="+", default=["walsh_degree_4", "full_ucr"])
    parser.add_argument("--min-train-sdp-fidelity", type=float, default=0.999999)
    parser.add_argument("--max-train-sdp-abs-diff", type=float, default=5e-7)
    return parser.parse_args()


def seed_pair_for_instance(*, n_sys: int, M: int, instance_id: int) -> tuple[int, int]:
    return (
        100000 * int(n_sys) + 1000 * int(M) + int(instance_id),
        200000 * int(n_sys) + 1000 * int(M) + int(instance_id),
    )


def n_anc_for_M(M: int) -> int:
    if int(M) < 2:
        raise ValueError("M must be >= 2.")
    return int(math.ceil(math.log2(int(M))))


def generate_haar_system_states(*, n_sys: int, M: int, state_seed: int, dtype: np.dtype) -> np.ndarray:
    sys_dim = 1 << int(n_sys)
    rng = np.random.Generator(np.random.PCG64(int(state_seed)))
    real = rng.normal(size=(int(M), sys_dim))
    imag = rng.normal(size=(int(M), sys_dim))
    states = real + 1j * imag
    norms = np.linalg.norm(states, axis=1, keepdims=True)
    if np.any(norms <= 1e-15):
        raise ValueError("Encountered near-zero norm while generating Haar states.")
    states = states / norms
    return states.astype(dtype, copy=False)


def embed_system_states(system_states: np.ndarray, *, n_anc: int) -> np.ndarray:
    M, sys_dim = system_states.shape
    anc_dim = 1 << int(n_anc)
    full_state = np.zeros((int(M), sys_dim * anc_dim), dtype=system_states.dtype)
    full_indices = np.arange(sys_dim, dtype=np.int64) * anc_dim
    full_state[:, full_indices] = system_states
    return full_state


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    hasher = hashlib.sha256()
    hasher.update(str(tuple(contiguous.shape)).encode("utf-8"))
    hasher.update(str(contiguous.dtype).encode("utf-8"))
    hasher.update(contiguous.tobytes())
    return hasher.hexdigest()


def fidelity_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a128 = np.asarray(a, dtype=np.complex128)
    b128 = np.asarray(b, dtype=np.complex128)
    a128 = a128 / np.linalg.norm(a128, axis=1, keepdims=True)
    b128 = b128 / np.linalg.norm(b128, axis=1, keepdims=True)
    fidelities = np.abs(np.sum(np.conj(a128) * b128, axis=1)) ** 2
    return {
        "min_fidelity": float(np.min(fidelities)),
        "max_fidelity_error": float(np.max(1.0 - fidelities)),
        "max_abs_diff": float(np.max(np.abs(a128 - b128))),
    }


def expected_training_state(*, n_sys: int, M: int, state_seed: int) -> np.ndarray:
    system = generate_haar_system_states(n_sys=n_sys, M=M, state_seed=state_seed, dtype=np.dtype(np.complex64))
    return embed_system_states(system, n_anc=n_anc_for_M(M))


def expected_sdp_system_state(*, n_sys: int, M: int, state_seed: int) -> np.ndarray:
    return generate_haar_system_states(n_sys=n_sys, M=M, state_seed=state_seed, dtype=np.dtype(np.complex128))


def scan_summary_paths(paths: Sequence[str]) -> list[Path]:
    summary_paths: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_file() and path.name.endswith("_summary.json"):
            summary_paths.append(path)
            continue
        direct = path / "summaries"
        if direct.exists():
            summary_paths.extend(sorted(direct.glob("*_summary.json")))
            continue
        summary_paths.extend(sorted(path.rglob("summaries/*_summary.json")))
    return sorted(set(summary_paths))


def load_summary(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    config = summary["config"]
    state = summary["state_precompute"]
    return {
        "path": str(path),
        "state_family": config.get("state_family", "unknown"),
        "n_sys": int(config["n_sys"]),
        "M": int(config["M"]),
        "instance_id": int(config["instance_id"]),
        "model_type": str(config["model_type"]),
        "state_seed": int(state["state_seed"]),
        "state_array_sha256": state.get("state_array_sha256"),
        "p_opt_sdp": config.get("p_opt_sdp"),
    }


def verify_summaries(summary_paths: Sequence[Path], args: argparse.Namespace) -> int:
    rows = [load_summary(path) for path in summary_paths]
    if not rows:
        print("No summary JSON files found.")
        return 1

    status = 0
    groups: dict[tuple[str, int, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["state_family"], row["n_sys"], row["M"], row["instance_id"])
        groups.setdefault(key, []).append(row)

    for key, group_rows in sorted(groups.items()):
        state_family, n_sys, M, instance_id = key
        seeds = {row["state_seed"] for row in group_rows}
        hashes = {row["state_array_sha256"] for row in group_rows}
        hashes.discard(None)
        p_opts = {row["p_opt_sdp"] for row in group_rows}
        models = ",".join(sorted(row["model_type"] for row in group_rows))

        group_ok = len(seeds) == 1 and len(hashes) <= 1 and len(p_opts) <= 1
        detail = ""
        if state_family == "haar" and len(seeds) == 1:
            state_seed = next(iter(seeds))
            train = expected_training_state(n_sys=n_sys, M=M, state_seed=state_seed)
            expected_hash = array_sha256(train)
            if hashes and hashes != {expected_hash}:
                group_ok = False
            sdp = expected_sdp_system_state(n_sys=n_sys, M=M, state_seed=state_seed)
            train_system = train[:, :: 1 << n_anc_for_M(M)]
            metrics = fidelity_metrics(train_system, sdp)
            if metrics["min_fidelity"] < float(args.min_train_sdp_fidelity):
                group_ok = False
            if metrics["max_abs_diff"] > float(args.max_train_sdp_abs_diff):
                group_ok = False
            detail = (
                f" train_sdp_min_fidelity={metrics['min_fidelity']:.12g}"
                f" train_sdp_max_abs_diff={metrics['max_abs_diff']:.3g}"
            )

        prefix = "OK" if group_ok else "FAIL"
        print(
            f"{prefix} family={state_family} n_sys={n_sys} M={M} instance={instance_id:02d}"
            f" models={models} seeds={sorted(seeds)} hashes={len(hashes)} p_opts={len(p_opts)}{detail}"
        )
        if not group_ok:
            status = 1
            for row in group_rows:
                print(
                    f"  {row['model_type']}: seed={row['state_seed']}"
                    f" hash={row['state_array_sha256']} p_opt_sdp={row['p_opt_sdp']}"
                    f" path={row['path']}"
                )
    return status


def verify_generated(args: argparse.Namespace) -> int:
    status = 0
    for instance_id in range(int(args.instance_start), int(args.instance_end) + 1):
        _, data_seed = seed_pair_for_instance(n_sys=int(args.n_sys), M=int(args.M), instance_id=instance_id)
        state_seed = int(data_seed)
        reference = expected_training_state(n_sys=int(args.n_sys), M=int(args.M), state_seed=state_seed)
        reference_hash = array_sha256(reference)
        model_hashes = {}
        min_model_fidelity = 1.0
        max_model_abs_diff = 0.0
        for model_type in args.models:
            candidate = expected_training_state(n_sys=int(args.n_sys), M=int(args.M), state_seed=state_seed)
            model_hashes[str(model_type)] = array_sha256(candidate)
            metrics = fidelity_metrics(reference, candidate)
            min_model_fidelity = min(min_model_fidelity, metrics["min_fidelity"])
            max_model_abs_diff = max(max_model_abs_diff, metrics["max_abs_diff"])

        sdp = expected_sdp_system_state(n_sys=int(args.n_sys), M=int(args.M), state_seed=state_seed)
        train_system = reference[:, :: 1 << n_anc_for_M(int(args.M))]
        train_sdp = fidelity_metrics(train_system, sdp)
        ok = (
            set(model_hashes.values()) == {reference_hash}
            and train_sdp["min_fidelity"] >= float(args.min_train_sdp_fidelity)
            and train_sdp["max_abs_diff"] <= float(args.max_train_sdp_abs_diff)
        )
        prefix = "OK" if ok else "FAIL"
        print(
            f"{prefix} n_sys={int(args.n_sys)} M={int(args.M)} instance={instance_id:02d}"
            f" state_seed={state_seed} model_hashes={len(set(model_hashes.values()))}"
            f" model_min_fidelity={min_model_fidelity:.12g}"
            f" model_max_abs_diff={max_model_abs_diff:.3g}"
            f" train_sdp_min_fidelity={train_sdp['min_fidelity']:.12g}"
            f" train_sdp_max_abs_diff={train_sdp['max_abs_diff']:.3g}"
        )
        if not ok:
            status = 1
    return status


def main() -> None:
    args = parse_args()
    if args.result_dirs:
        raise SystemExit(verify_summaries(scan_summary_paths(args.result_dirs), args))
    raise SystemExit(verify_generated(args))


if __name__ == "__main__":
    main()
