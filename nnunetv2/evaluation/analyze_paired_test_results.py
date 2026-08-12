#!/usr/bin/env python3
"""Paired case and lesion-volume analysis for two nnU-Net test summaries."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Mapping, Sequence

import SimpleITK as sitk


METRIC_NAMES = ("Dice", "IoU", "TP", "FP", "FN", "n_pred", "n_ref")
VOLUME_GROUPS = (
    ("lt_1ml", 0.0, 1.0),
    ("1_to_5ml", 1.0, 5.0),
    ("5_to_20ml", 5.0, 20.0),
    ("ge_20ml", 20.0, math.inf),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two independent-test summary.json files case by case and "
            "stratify results by reference lesion volume."
        )
    )
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--label", default="1")
    parser.add_argument("--top-k", default=10, type=int)
    return parser.parse_args()


def load_summary(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Summary does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if not isinstance(summary.get("metric_per_case"), list):
        raise RuntimeError(f"Invalid nnU-Net summary: {path}")
    return summary


def case_id(case: Mapping) -> str:
    reference_file = case.get("reference_file")
    if not reference_file:
        raise RuntimeError("A metric_per_case entry has no reference_file")
    name = Path(reference_file).name
    return name[: -len(".nii.gz")] if name.endswith(".nii.gz") else Path(name).stem


def index_cases(summary: Mapping, label: str) -> Dict[str, dict]:
    indexed = {}
    for case in summary["metric_per_case"]:
        identifier = case_id(case)
        if identifier in indexed:
            raise RuntimeError(f"Duplicate case in summary: {identifier}")
        metrics = case.get("metrics", {}).get(label)
        if metrics is None:
            available = sorted(case.get("metrics", {}))
            raise RuntimeError(
                f"Label {label!r} missing for {identifier}; available labels: {available}"
            )
        missing = [name for name in METRIC_NAMES if name not in metrics]
        if missing:
            raise RuntimeError(f"Metrics missing for {identifier}: {missing}")
        indexed[identifier] = case
    return indexed


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else math.nan


def mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return sum(values) / len(values) if values else math.nan


def volume_group(volume_ml: float) -> str:
    for name, lower, upper in VOLUME_GROUPS:
        if lower <= volume_ml < upper:
            return name
    raise RuntimeError(f"Unable to assign volume group for {volume_ml}")


def get_spacing_mm(reference_file: str) -> Sequence[float]:
    image = sitk.ReadImage(reference_file)
    spacing = tuple(float(value) for value in image.GetSpacing())
    if not spacing or any(value <= 0 for value in spacing):
        raise RuntimeError(f"Invalid spacing {spacing}: {reference_file}")
    return spacing


def model_metrics(metrics: Mapping) -> dict:
    tp = float(metrics["TP"])
    fp = float(metrics["FP"])
    fn = float(metrics["FN"])
    return {
        **{name: float(metrics[name]) for name in METRIC_NAMES},
        "Precision": safe_divide(tp, tp + fp),
        "Recall": safe_divide(tp, tp + fn),
    }


def build_case_rows(candidate: Mapping, baseline: Mapping, label: str) -> List[dict]:
    candidate_cases = index_cases(candidate, label)
    baseline_cases = index_cases(baseline, label)
    candidate_ids = set(candidate_cases)
    baseline_ids = set(baseline_cases)
    if candidate_ids != baseline_ids:
        raise RuntimeError(
            "Candidate and baseline case sets differ; "
            f"candidate_only={sorted(candidate_ids - baseline_ids)}, "
            f"baseline_only={sorted(baseline_ids - candidate_ids)}"
        )

    rows = []
    for identifier in sorted(candidate_ids):
        candidate_case = candidate_cases[identifier]
        baseline_case = baseline_cases[identifier]
        candidate_reference = Path(candidate_case["reference_file"]).resolve()
        baseline_reference = Path(baseline_case["reference_file"]).resolve()
        if candidate_reference != baseline_reference:
            raise RuntimeError(
                f"Reference paths differ for {identifier}: "
                f"{candidate_reference} != {baseline_reference}"
            )

        candidate_metrics = model_metrics(candidate_case["metrics"][label])
        baseline_metrics = model_metrics(baseline_case["metrics"][label])
        if candidate_metrics["n_ref"] != baseline_metrics["n_ref"]:
            raise RuntimeError(f"Reference voxel counts differ for {identifier}")

        spacing_mm = get_spacing_mm(str(candidate_reference))
        voxel_volume_mm3 = math.prod(spacing_mm)
        reference_volume_ml = candidate_metrics["n_ref"] * voxel_volume_mm3 / 1000.0
        row = {
            "case_id": identifier,
            "reference_file": str(candidate_reference),
            "spacing_mm": "x".join(f"{value:g}" for value in spacing_mm),
            "voxel_volume_mm3": voxel_volume_mm3,
            "reference_volume_ml": reference_volume_ml,
            "volume_group": volume_group(reference_volume_ml),
        }
        for prefix, metrics in (
            ("candidate", candidate_metrics),
            ("baseline", baseline_metrics),
        ):
            for name, value in metrics.items():
                row[f"{prefix}_{name}"] = value
        for name in (*METRIC_NAMES, "Precision", "Recall"):
            row[f"delta_{name}"] = candidate_metrics[name] - baseline_metrics[name]
        rows.append(row)
    return rows


def summarize_model(rows: Sequence[Mapping], prefix: str) -> dict:
    tp_sum = sum(float(row[f"{prefix}_TP"]) for row in rows)
    fp_sum = sum(float(row[f"{prefix}_FP"]) for row in rows)
    fn_sum = sum(float(row[f"{prefix}_FN"]) for row in rows)
    return {
        "Dice_mean": mean(row[f"{prefix}_Dice"] for row in rows),
        "Dice_median": median(float(row[f"{prefix}_Dice"]) for row in rows),
        "IoU_mean": mean(row[f"{prefix}_IoU"] for row in rows),
        "Precision_macro": mean(row[f"{prefix}_Precision"] for row in rows),
        "Recall_macro": mean(row[f"{prefix}_Recall"] for row in rows),
        "Dice_micro": safe_divide(2.0 * tp_sum, 2.0 * tp_sum + fp_sum + fn_sum),
        "Precision_micro": safe_divide(tp_sum, tp_sum + fp_sum),
        "Recall_micro": safe_divide(tp_sum, tp_sum + fn_sum),
        "TP_mean": mean(row[f"{prefix}_TP"] for row in rows),
        "FP_mean": mean(row[f"{prefix}_FP"] for row in rows),
        "FN_mean": mean(row[f"{prefix}_FN"] for row in rows),
        "TP_sum": tp_sum,
        "FP_sum": fp_sum,
        "FN_sum": fn_sum,
    }


def summarize_group(name: str, rows: Sequence[Mapping]) -> dict:
    deltas = [float(row["delta_Dice"]) for row in rows]
    tolerance = 1e-12
    return {
        "group": name,
        "case_count": len(rows),
        "volume_ml_min": min(float(row["reference_volume_ml"]) for row in rows),
        "volume_ml_max": max(float(row["reference_volume_ml"]) for row in rows),
        "volume_ml_mean": mean(row["reference_volume_ml"] for row in rows),
        "volume_ml_median": median(float(row["reference_volume_ml"]) for row in rows),
        "candidate": summarize_model(rows, "candidate"),
        "baseline": summarize_model(rows, "baseline"),
        "paired_delta": {
            "Dice_mean": mean(deltas),
            "Dice_median": median(deltas),
            "Dice_min": min(deltas),
            "Dice_max": max(deltas),
            "FP_mean": mean(row["delta_FP"] for row in rows),
            "FN_mean": mean(row["delta_FN"] for row in rows),
            "FP_sum": sum(float(row["delta_FP"]) for row in rows),
            "FN_sum": sum(float(row["delta_FN"]) for row in rows),
            "wins": sum(delta > tolerance for delta in deltas),
            "ties": sum(abs(delta) <= tolerance for delta in deltas),
            "losses": sum(delta < -tolerance for delta in deltas),
        },
    }


def build_analysis(rows: Sequence[Mapping], top_k: int) -> dict:
    groups = {"all": summarize_group("all", rows)}
    for name, _, _ in VOLUME_GROUPS:
        selected = [row for row in rows if row["volume_group"] == name]
        if not selected:
            raise RuntimeError(f"No cases in expected volume group: {name}")
        groups[name] = summarize_group(name, selected)

    ranked = sorted(rows, key=lambda row: float(row["delta_Dice"]))
    fields = (
        "case_id",
        "reference_volume_ml",
        "volume_group",
        "candidate_Dice",
        "baseline_Dice",
        "delta_Dice",
        "delta_FP",
        "delta_FN",
    )

    def select_fields(row: Mapping) -> dict:
        return {field: row[field] for field in fields}

    return {
        "case_count": len(rows),
        "volume_thresholds_ml": [
            {
                "group": name,
                "lower_inclusive": lower,
                "upper_exclusive": upper if math.isfinite(upper) else None,
            }
            for name, lower, upper in VOLUME_GROUPS
        ],
        "groups": groups,
        "largest_dice_losses": [select_fields(row) for row in ranked[:top_k]],
        "largest_dice_gains": [select_fields(row) for row in reversed(ranked[-top_k:])],
    }


def flatten_group(group: Mapping) -> dict:
    row = {
        key: value
        for key, value in group.items()
        if key not in {"candidate", "baseline", "paired_delta"}
    }
    for section in ("candidate", "baseline", "paired_delta"):
        row.update({f"{section}_{key}": value for key, value in group[section].items()})
    return row


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_number(value: float, digits: int = 5) -> str:
    return "nan" if not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def print_report(analysis: Mapping, output_dir: Path) -> None:
    print("=" * 116)
    print("PAIRED INDEPENDENT-TEST ANALYSIS")
    print("=" * 116)
    print(
        f"{'group':<12} {'n':>4} {'candidate':>11} {'baseline':>11} "
        f"{'delta':>10} {'W/T/L':>11} {'delta FP':>12} {'delta FN':>12}"
    )
    print("-" * 116)
    for name in ("all", *(item[0] for item in VOLUME_GROUPS)):
        group = analysis["groups"][name]
        delta = group["paired_delta"]
        print(
            f"{name:<12} {group['case_count']:>4d} "
            f"{format_number(group['candidate']['Dice_mean']):>11} "
            f"{format_number(group['baseline']['Dice_mean']):>11} "
            f"{format_number(delta['Dice_mean'], 6):>10} "
            f"{delta['wins']}/{delta['ties']}/{delta['losses']:>3} "
            f"{format_number(delta['FP_mean'], 2):>12} "
            f"{format_number(delta['FN_mean'], 2):>12}"
        )
    print("-" * 116)
    print(f"Analysis files: {output_dir}")
    print("=" * 116)


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    candidate = load_summary(args.candidate.resolve())
    baseline = load_summary(args.baseline.resolve())
    rows = build_case_rows(candidate, baseline, args.label)
    analysis = build_analysis(rows, min(args.top_k, len(rows)))
    analysis.update(
        {
            "candidate_summary": str(args.candidate.resolve()),
            "baseline_summary": str(args.baseline.resolve()),
            "label": args.label,
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "paired_cases.csv", rows)
    group_rows = [flatten_group(group) for group in analysis["groups"].values()]
    write_csv(args.output_dir / "volume_strata.csv", group_rows)
    with (args.output_dir / "analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print_report(analysis, args.output_dir.resolve())


if __name__ == "__main__":
    main()
