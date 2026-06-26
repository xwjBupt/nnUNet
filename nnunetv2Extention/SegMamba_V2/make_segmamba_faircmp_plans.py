#!/usr/bin/env python3
"""
Create a SegMamba fairness-comparison plans file for a given nnU-Net dataset.

The generated plans keep the 3d_fullres spatial setup and only swap the network
architecture to SegMamba. This is intended to isolate the backbone effect from
the preprocessing effect.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from batchgenerators.utilities.file_and_folder_operations import load_json, maybe_mkdir_p, save_json

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name


def build_faircmp_config(base_config: dict, reuse_preprocessed: bool) -> dict:
    cfg = deepcopy(base_config)
    cfg["data_identifier"] = "nnUNetPlans_3d_fullres" if reuse_preprocessed else "nnUNetPlans_segmamba_faircmp"
    cfg["batch_size"] = 2
    cfg["batch_dice"] = False
    cfg["architecture"] = {
        "network_class_name": "nnunetv2Extention.SegMamba_V2.segmambav2.SegMamba",
        "arch_kwargs": {
            "depths": [2, 2, 2, 2],
            "feat_size": [48, 96, 192, 384],
            "drop_path_rate": 0.0,
            "layer_scale_init_value": 1e-6,
            "hidden_size": 768,
            "norm_name": "instance",
            "conv_block": True,
            "res_block": True,
            "spatial_dims": 3,
            "deep_supervision": True,
            "pretrained_path": None,
            "strides": [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        },
        "_kw_requires_import": [],
    }
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="515", help="Dataset ID or dataset name.")
    parser.add_argument(
        "--source-plans-name",
        default="nnUNetPlans",
        help="Existing plans identifier to clone from. Default: nnUNetPlans",
    )
    parser.add_argument(
        "--output-plans-name",
        default="nnUNetPlans_segmamba_faircmp",
        help="Plans identifier of the generated file. Default: nnUNetPlans_segmamba_faircmp",
    )
    parser.add_argument(
        "--config-name",
        default="segmamba_faircmp",
        help="Configuration name to add to the output plans. Default: segmamba_faircmp",
    )
    parser.add_argument(
        "--reuse-preprocessed",
        action="store_true",
        help="Reuse the existing nnUNetPlans_3d_fullres preprocessed data folder instead of creating a new one.",
    )
    args = parser.parse_args()

    dataset_name = maybe_convert_to_dataset_name(args.dataset)
    dataset_dir = Path(str(nnUNet_preprocessed)) / dataset_name
    source_plans_file = dataset_dir / f"{args.source_plans_name}.json"
    output_plans_file = dataset_dir / f"{args.output_plans_name}.json"

    if not source_plans_file.is_file():
        raise FileNotFoundError(f"Missing source plans file: {source_plans_file}")

    plans = load_json(source_plans_file)
    if "3d_fullres" not in plans.get("configurations", {}):
        raise KeyError("The source plans file does not contain a 3d_fullres configuration.")

    new_plans = deepcopy(plans)
    new_plans["plans_name"] = args.output_plans_name
    new_plans["configurations"][args.config_name] = build_faircmp_config(
        plans["configurations"]["3d_fullres"],
        reuse_preprocessed=args.reuse_preprocessed,
    )

    maybe_mkdir_p(dataset_dir)
    save_json(new_plans, output_plans_file, sort_keys=False)
    print(f"Wrote {output_plans_file}")
    print(f"Use: nnUNetv2_train {dataset_name} {args.config_name} all -tr nnUNetTrainerSegMamba -p {args.output_plans_name}")


if __name__ == "__main__":
    main()
