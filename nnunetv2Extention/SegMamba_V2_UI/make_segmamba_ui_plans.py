#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from batchgenerators.utilities.file_and_folder_operations import load_json, maybe_mkdir_p, save_json

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name


def build_ui_config(base_config: dict) -> dict:
    cfg = deepcopy(base_config)
    cfg["architecture"] = deepcopy(cfg["architecture"])
    cfg["architecture"]["network_class_name"] = "nnunetv2Extention.SegMamba_V2_UI.segmambav2_ui.SegMamba"
    cfg["architecture"]["arch_kwargs"] = deepcopy(cfg["architecture"].get("arch_kwargs", {}))
    cfg["architecture"]["arch_kwargs"]["deep_supervision"] = True
    cfg["architecture"].setdefault("_kw_requires_import", [])
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="515", help="Dataset ID or dataset name.")
    parser.add_argument("--source-plans-name", default="nnUNetPlans")
    parser.add_argument("--source-config-name", default="segmamba")
    parser.add_argument("--output-plans-name", default="nnUNetPlans_segmamba_ui")
    parser.add_argument("--config-name", default="segmamba_ui")
    args = parser.parse_args()

    dataset_name = maybe_convert_to_dataset_name(args.dataset)
    dataset_dir = Path(str(nnUNet_preprocessed)) / dataset_name
    source_plans_file = dataset_dir / f"{args.source_plans_name}.json"
    output_plans_file = dataset_dir / f"{args.output_plans_name}.json"

    if not source_plans_file.is_file():
        raise FileNotFoundError(f"Missing source plans file: {source_plans_file}")

    source_plans = load_json(source_plans_file)
    if args.source_config_name not in source_plans.get("configurations", {}):
        raise KeyError(
            f"Missing source config {args.source_config_name}. "
            f"Available: {list(source_plans.get('configurations', {}).keys())}"
        )

    new_plans = load_json(output_plans_file) if output_plans_file.is_file() else deepcopy(source_plans)
    new_plans["plans_name"] = args.output_plans_name
    new_plans["configurations"][args.config_name] = build_ui_config(
        source_plans["configurations"][args.source_config_name]
    )

    maybe_mkdir_p(dataset_dir)
    save_json(new_plans, output_plans_file, sort_keys=False)
    print(f"Wrote {output_plans_file}")
    print(f"Use: nnUNetv2_train {dataset_name} {args.config_name} all -tr nnUNetTrainerSegMambaUI -p {args.output_plans_name}")


if __name__ == "__main__":
    main()
