from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kineroute_nvp.evaluation.plots import plot_validation_comparison


@dataclass(frozen=True)
class RunCurve:
    directory: Path
    label: str
    seed: int | None
    validation_split: str
    validation_samples: int | None
    dataset_roots: tuple[str, ...]
    epochs: list[int]
    val_ade: list[float]
    val_fde: list[float]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine validation ADE/FDE histories from explicit result folders into one figure."
    )
    parser.add_argument(
        "result_folders",
        nargs="+",
        help="Result folder paths, or folder names resolved below --results-root.",
    )
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output", default="results/validation_curves_combined.png")
    parser.add_argument("--title", default="Validation Metrics by Epoch")
    return parser.parse_args(argv)


def _resolve_run_dir(value: str, results_root: Path) -> Path:
    supplied = Path(value).expanduser()
    candidates = (supplied, results_root / supplied)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Result folder not found: {value!r} (also tried {results_root / supplied})"
    )


def _prettify_run_name(run_name: str) -> str:
    cleaned = run_name
    for suffix in ("_paper_clean", "_paper", "_coords", "_clean"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
    aliases = {
        "seq2seq": "Seq2Seq",
        "gru": "GRU",
        "bigru": "Bi-GRU",
        "lstm": "LSTM",
        "bilstm": "Bi-LSTM",
        "kine_real_nvp": "Kine-Real-NVP",
    }
    return aliases.get(cleaned, cleaned.replace("_", " "))


def _load_payload(run_dir: Path) -> dict[str, Any]:
    results_path = run_dir / "_results.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"Missing {results_path}")
    payload = json.loads(results_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {results_path}")
    return payload


def _load_history(run_dir: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    history = payload.get("history")
    if isinstance(history, list) and history:
        return history

    metrics_path = run_dir / "logs" / "metrics.jsonl"
    if metrics_path.is_file():
        with metrics_path.open() as handle:
            return [json.loads(line) for line in handle if line.strip()]
    raise ValueError(f"No validation history found in {run_dir}")


def load_run_curve(run_dir: Path) -> RunCurve:
    payload = _load_payload(run_dir)
    history = _load_history(run_dir, payload)
    epochs: list[int] = []
    val_ade: list[float] = []
    val_fde: list[float] = []
    for record in history:
        validation = record.get("val", {})
        ade_value = validation.get("ade", record.get("val_ade"))
        fde_value = validation.get("fde", record.get("val_fde"))
        if ade_value is None or fde_value is None:
            raise ValueError(f"Epoch record lacks validation ADE/FDE in {run_dir}: {record}")
        epochs.append(int(record["epoch"]))
        val_ade.append(float(ade_value))
        val_fde.append(float(fde_value))

    config = payload.get("resolved_configuration", {})
    data_config = config.get("data", {})
    dataset_paths = payload.get("dataset_paths", {})
    raw_roots = dataset_paths.get("raw_roots") or data_config.get("raw_roots") or []
    if isinstance(raw_roots, str):
        raw_roots = [raw_roots]
    split_sizes = payload.get("dataset_split_sizes", {})
    run_name = str(payload.get("run_name") or run_dir.name.split("__", 2)[-1])
    seed = payload.get("seed", config.get("seed"))
    return RunCurve(
        directory=run_dir,
        label=_prettify_run_name(run_name),
        seed=int(seed) if seed is not None else None,
        validation_split=str(data_config.get("val_split", "val")),
        validation_samples=int(split_sizes["val"]) if "val" in split_sizes else None,
        dataset_roots=tuple(sorted(map(str, raw_roots))),
        epochs=epochs,
        val_ade=val_ade,
        val_fde=val_fde,
    )


def _unique_labels(curves: list[RunCurve]) -> list[str]:
    counts: dict[str, int] = {}
    for curve in curves:
        counts[curve.label] = counts.get(curve.label, 0) + 1
    return [
        f"{curve.label} (seed={curve.seed})" if counts[curve.label] > 1 else curve.label
        for curve in curves
    ]


def _report_compatibility(curves: list[RunCurve]) -> None:
    seeds = sorted({curve.seed for curve in curves}, key=lambda value: (value is None, value))
    split_names = sorted({curve.validation_split for curve in curves})
    split_sizes = sorted({curve.validation_samples for curve in curves}, key=lambda value: (value is None, value))
    dataset_roots = {curve.dataset_roots for curve in curves}
    print(f"Seeds: {seeds}")
    print(f"Validation split names: {split_names}")
    print(f"Validation sample counts: {split_sizes}")
    if len(split_names) > 1 or len(split_sizes) > 1 or len(dataset_roots) > 1:
        print(
            "WARNING: result folders do not describe the same validation dataset; "
            "the curves may not be directly comparable.",
            file=sys.stderr,
        )
    if len(seeds) > 1:
        print(
            "NOTE: seeds differ. The split is still fixed, but these are different "
            "stochastic training replicates.",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results_root = Path(args.results_root).expanduser()
    curves = [load_run_curve(_resolve_run_dir(value, results_root)) for value in args.result_folders]
    _report_compatibility(curves)
    labels = _unique_labels(curves)
    output_path = Path(args.output).expanduser().resolve()
    plot_validation_comparison(
        [
            {
                "label": label,
                "epochs": curve.epochs,
                "ade": curve.val_ade,
                "fde": curve.val_fde,
            }
            for curve, label in zip(curves, labels)
        ],
        output_path,
        title=args.title,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
