#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


DEFAULT_PREPROCESSED = "/home/wjx/CodeData/data/nnUNetData/nnUNet_preprocessed"
PLANS_NAME = "nnUNetPlans_segmamba_ui"
PARENT_CONFIGURATION = (
    "segmamba_uig_dec2_logit_boundary_hierarchy_loss_128x96x96"
)
CONFIGURATION_NAME = (
    "segmamba_uig_dec2_logit_boundary_hierarchy_highres2_reliable_128x96x96"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add the HighRes Target-Reliable hierarchy UIG experiment."
    )
    parser.add_argument("--dataset-name", default="Dataset515_ICH2023")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--preprocessed-root",
        default=os.environ.get("nnUNet_preprocessed", DEFAULT_PREPROCESSED),
    )
    args = parser.parse_args()

    dataset_dir = Path(args.preprocessed_root) / args.dataset_name
    plans_path = dataset_dir / f"{PLANS_NAME}.json"
    data_dir = dataset_dir / "nnUNetPlans_segmamba"
    if not plans_path.is_file():
        raise FileNotFoundError(f"Missing plans file: {plans_path}")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Missing preprocessed data: {data_dir}")

    with plans_path.open("r", encoding="utf-8") as file:
        plans = json.load(file)

    configurations = plans.get("configurations")
    if not isinstance(configurations, dict):
        raise RuntimeError(f"No configurations object in {plans_path}")
    if PARENT_CONFIGURATION not in configurations:
        raise KeyError(
            f"Parent configuration {PARENT_CONFIGURATION!r} is missing from {plans_path}"
        )

    configurations[CONFIGURATION_NAME] = {
        "inherits_from": PARENT_CONFIGURATION,
        "batch_size": int(args.batch_size),
    }

    temporary_path = plans_path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(plans, file, indent=4)
        file.write("\n")
    temporary_path.replace(plans_path)

    print(f"Validated preprocessed data: {data_dir}")
    print(f"Updated plans: {plans_path}")
    print(f"Parent config: {PARENT_CONFIGURATION}")
    print(f"New config: {CONFIGURATION_NAME}")
    print(f"Total batch size: {args.batch_size}")


if __name__ == "__main__":
    main()
