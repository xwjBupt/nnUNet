#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from batchgenerators.utilities.file_and_folder_operations import load_json, maybe_mkdir_p, save_json

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name


PRETRAINED_PATH = "/home/wjx/CodeData/code/nnUNet/nnunetv2Extention/SegMamba_V2/final_model_0.8388.pt"


def build_segmamba_config(base_config: dict) -> dict:
    cfg = deepcopy(base_config)
    cfg["data_identifier"] = "nnUNetPlans_segmamba"
    cfg["spacing"] = [1.0, 1.0, 1.0]
    cfg["patch_size"] = [128, 128, 128]
    cfg["median_image_size_in_voxels"] = [128, 128, 128]
    cfg["batch_size"] = 2
    cfg["batch_dice"] = True
    cfg["architecture"] = {
        "network_class_name": "nnunetv2Extention.SegMamba_V2.segmambav2.SegMamba",
        "arch_kwargs": {
            "deep_supervision": True,
            "pretrained_path": PRETRAINED_PATH,
            "strides": [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        },
        "_kw_requires_import": [],
    }
    return cfg


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
    source_config = source_plans.get("configurations", {}).get(args.source_config_name)
    if source_config is None:
        if "3d_fullres" not in source_plans.get("configurations", {}):
            raise KeyError(
                f"Missing source config {args.source_config_name} and fallback 3d_fullres. "
                f"Available: {list(source_plans.get('configurations', {}).keys())}"
            )
        source_config = build_segmamba_config(source_plans["configurations"]["3d_fullres"])

    new_plans = load_json(output_plans_file) if output_plans_file.is_file() else deepcopy(source_plans)
    new_plans["plans_name"] = args.output_plans_name
    new_plans["configurations"][args.source_config_name] = source_config
    new_plans["configurations"][args.config_name] = build_ui_config(source_config)

    maybe_mkdir_p(dataset_dir)
    save_json(new_plans, output_plans_file, sort_keys=False)
    print(f"Wrote {output_plans_file}")
    print(f"Use: nnUNetv2_train {dataset_name} {args.config_name} all -tr nnUNetTrainerSegMambaUI -p {args.output_plans_name}")


if __name__ == "__main__":
    main()
