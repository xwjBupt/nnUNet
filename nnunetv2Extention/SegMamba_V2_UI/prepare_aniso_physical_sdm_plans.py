#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path


DEFAULT_PREPROCESSED = "/home/wjx/CodeData/data/nnUNetData/nnUNet_preprocessed"
PRETRAINED_PATH = "/home/wjx/CodeData/code/nnUNet/nnunetv2Extention/SegMamba_V2/final_model_0.8388.pt"
CONFIG_NAME = "segmamba_ui_aniso_physical_sdmxy_16x192x192"
PLANS_NAME = "nnUNetPlans_segmamba_ui_aniso"


def build_configuration(source: dict, batch_size: int) -> dict:
    configuration = deepcopy(source)
    configuration.update(
        {
            "data_identifier": "nnUNetPlans_3d_fullres",
            "batch_size": int(batch_size),
            "patch_size": [16, 192, 192],
            "median_image_size_in_voxels": [28.0, 512.0, 512.0],
            "spacing": [5.0, 0.4882810115814209, 0.4882810115814209],
            "batch_dice": True,
            "physical_band_radius_mm": 2.0,
            "architecture": {
                "network_class_name": (
                    "nnunetv2Extention.SegMamba_V2_UI.segmambav2_ui.SegMamba"
                ),
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
                    "pretrained_path": PRETRAINED_PATH,
                    "strides": [[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]],
                    "encoder_kernel_sizes": [
                        [1, 3, 3],
                        [1, 3, 3],
                        [3, 3, 3],
                        [3, 3, 3],
                        [3, 3, 3],
                    ],
                    "sdm_level": "dec1",
                    "sdm_residual_scale_limit": 0.1,
                },
                "_kw_requires_import": [],
            },
        }
    )
    return configuration


def count_native_cases(native_data_dir: Path) -> tuple[int, int]:
    data_cases = {
        path.name.removesuffix(".b2nd")
        for path in native_data_dir.glob("*.b2nd")
        if not path.name.endswith("_seg.b2nd")
    }
    segmentation_cases = {
        path.name.removesuffix("_seg.b2nd")
        for path in native_data_dir.glob("*_seg.b2nd")
    }
    return len(data_cases), len(segmentation_cases)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Dataset515 native-spacing SegMamba UI + physical band + SDM-XY plans."
    )
    parser.add_argument("--dataset-name", default="Dataset515_ICH2023")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--preprocessed-root",
        default=os.environ.get("nnUNet_preprocessed", DEFAULT_PREPROCESSED),
    )
    args = parser.parse_args()

    dataset_dir = Path(args.preprocessed_root) / args.dataset_name
    source_plans_path = dataset_dir / "nnUNetPlans.json"
    output_plans_path = dataset_dir / f"{PLANS_NAME}.json"
    native_data_dir = dataset_dir / "nnUNetPlans_3d_fullres"

    if not source_plans_path.is_file():
        raise FileNotFoundError(f"Missing source plans: {source_plans_path}")
    if not native_data_dir.is_dir():
        raise FileNotFoundError(
            f"Missing native-spacing preprocessed data: {native_data_dir}. "
            "Run nnUNetv2_plan_and_preprocess for 3d_fullres first."
        )

    data_count, segmentation_count = count_native_cases(native_data_dir)
    if data_count == 0 or data_count != segmentation_count:
        raise RuntimeError(
            "Incomplete native-spacing data: "
            f"images={data_count}, segmentations={segmentation_count}, folder={native_data_dir}"
        )

    with source_plans_path.open("r", encoding="utf-8") as file:
        plans = json.load(file)
    source_configuration = plans.get("configurations", {}).get("3d_fullres")
    if source_configuration is None:
        raise KeyError(f"3d_fullres is missing from {source_plans_path}")

    plans["plans_name"] = PLANS_NAME
    plans["configurations"][CONFIG_NAME] = build_configuration(
        source_configuration, args.batch_size
    )

    temporary_path = output_plans_path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(plans, file, indent=4)
        file.write("\n")
    temporary_path.replace(output_plans_path)

    print(f"Validated native data: {data_count} image/segmentation pairs in {native_data_dir}")
    print(f"Wrote plans: {output_plans_path}")
    print(f"Plans/config: {PLANS_NAME} / {CONFIG_NAME}")


if __name__ == "__main__":
    main()

