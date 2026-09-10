"""Shared Dataset515 utilities for external PyTorch model adapters.

The adapters in external repositories deliberately use this module instead of
the nnUNet trainer.  It keeps the data protocol identical to the UIGNet
experiments while leaving each upstream network and objective in its own repo.
"""

from __future__ import annotations

import csv
import fcntl
import json
import math
import os
import random
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


NNUNET_ROOT = Path(os.environ.get("NNUNET_ROOT", "/home/wjx/CodeData/code/nnUNet"))
DATA_ROOT = Path(os.environ.get("NNUNET_DATA_ROOT", "/home/wjx/CodeData/data/nnUNetData"))
RAW_ROOT = DATA_ROOT / "nnUNet_raw"
PREPROCESSED_ROOT = DATA_ROOT / "nnUNet_preprocessed"
DATASET_NAME = "Dataset515_ICH2023"
TRAIN_PLAN = "nnUNetPlans_segmamba"
TEST_PLAN = "nnUNetPlans_segmamba_ui"
TEST_CONFIGURATION = (
    "segmamba_uig_dec2_logit_boundary_hierarchy_core_exterior_masked_"
    "foreground_sample_dice_128x96x96"
)
PATCH_SIZE = (128, 96, 96)
RESULTS_ROOT = NNUNET_ROOT / "nnUNet_results" / DATASET_NAME
NNUNET_PYTHON = Path(
    os.environ.get("NNUNET_PYTHON", "/home/wjx/miniconda3/envs/nnunet_seg/bin/python")
)


def setup_distributed() -> tuple[int, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the Dataset515 adapters.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return rank, world_size, local_rank, torch.device("cuda", local_rank)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Dataset515 adapters.")
    return 0, 1, 0, torch.device("cuda", 0)


def barrier(world_size: int, device: torch.device) -> None:
    if world_size > 1:
        dist.barrier(device_ids=[device.index])


def set_seed(seed: int, rank: int) -> None:
    value = int(seed) + int(rank)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    from torch.nn.parallel import DistributedDataParallel

    return model.module if isinstance(model, DistributedDataParallel) else model


def json_args(args) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


class Dataset515PatchDataset(Dataset):
    """Foreground-aware random 3D patches from the shared 1 mm arrays."""

    def __init__(
        self,
        data_folder: Path,
        patch_size: Sequence[int] = PATCH_SIZE,
        samples_per_epoch: int = 1,
        foreground_sampling_probability: float = 1.0 / 3.0,
        seed: int = 515,
    ) -> None:
        self.data_folder = str(data_folder)
        self.patch_size = tuple(int(i) for i in patch_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.foreground_sampling_probability = float(foreground_sampling_probability)
        self.seed = int(seed)
        self.epoch = 0
        self._dataset = None
        self._identifiers = None

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _open(self):
        if self._dataset is None:
            from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2

            self._identifiers = sorted(
                nnUNetDatasetBlosc2.get_identifiers(self.data_folder)
            )
            self._dataset = nnUNetDatasetBlosc2(
                self.data_folder, identifiers=self._identifiers
            )
        return self._dataset

    @staticmethod
    def extract(array, lower: Sequence[int], patch_size: Sequence[int]) -> np.ndarray:
        shape = tuple(int(i) for i in array.shape[1:])
        slices = []
        padding = [(0, 0)]
        for lo, size, dim in zip(lower, patch_size, shape):
            hi = int(lo) + int(size)
            source_lo = max(int(lo), 0)
            source_hi = min(hi, dim)
            slices.append(slice(source_lo, source_hi))
            padding.append((source_lo - int(lo), hi - source_hi))
        patch = np.asarray(array[(slice(None), *slices)])
        if any(before or after for before, after in padding[1:]):
            patch = np.pad(patch, padding, mode="constant")
        expected = tuple(int(i) for i in patch_size)
        if tuple(patch.shape[1:]) != expected:
            raise RuntimeError(f"Expected (*, {expected}), got {patch.shape}.")
        return np.ascontiguousarray(patch)

    @staticmethod
    def random_lower(shape: Sequence[int], patch_size: Sequence[int], rng) -> tuple[int, ...]:
        lower = []
        for dim, size in zip(shape, patch_size):
            if int(dim) <= int(size):
                lower.append((int(dim) - int(size)) // 2)
            else:
                lower.append(int(rng.integers(0, int(dim) - int(size) + 1)))
        return tuple(lower)

    @staticmethod
    def foreground_lower(
        shape: Sequence[int], patch_size: Sequence[int], center: Sequence[int]
    ) -> tuple[int, ...]:
        lower = []
        for dim, size, coordinate in zip(shape, patch_size, center):
            if int(dim) <= int(size):
                lower.append((int(dim) - int(size)) // 2)
            else:
                start = int(coordinate) - int(size) // 2
                lower.append(max(0, min(start, int(dim) - int(size))))
        return tuple(lower)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        dataset = self._open()
        rng = np.random.default_rng(
            self.seed + self.epoch * 1_000_003 + int(index) * 97
        )
        identifier = self._identifiers[int(rng.integers(0, len(self._identifiers)))]
        data, segmentation, _, properties = dataset.load_case(identifier)
        shape = tuple(int(i) for i in data.shape[1:])
        locations = properties.get("class_locations", {}).get(1, [])
        use_foreground = (
            len(locations) > 0
            and rng.random() < self.foreground_sampling_probability
        )
        if use_foreground:
            location = locations[int(rng.integers(0, len(locations)))]
            lower = self.foreground_lower(shape, self.patch_size, location[-3:])
        else:
            lower = self.random_lower(shape, self.patch_size, rng)
        data_patch = self.extract(data, lower, self.patch_size).astype(np.float32)
        target_patch = (self.extract(segmentation, lower, self.patch_size) > 0).astype(
            np.int64
        )
        for axis in range(1, 4):
            if rng.random() < 0.5:
                data_patch = np.flip(data_patch, axis=axis).copy()
                target_patch = np.flip(target_patch, axis=axis).copy()
        return {
            "data": torch.from_numpy(data_patch),
            "target": torch.from_numpy(target_patch[0]),
        }


class Dataset515SlicePatchDataset(Dataset):
    """2.5D CENet samples: three neighboring slices, center-slice target."""

    def __init__(
        self,
        data_folder: Path,
        samples_per_epoch: int,
        foreground_sampling_probability: float,
        seed: int,
        patch_hw: Sequence[int] = (96, 96),
    ) -> None:
        self.data_folder = str(data_folder)
        self.samples_per_epoch = int(samples_per_epoch)
        self.foreground_sampling_probability = float(foreground_sampling_probability)
        self.seed = int(seed)
        self.patch_hw = tuple(int(i) for i in patch_hw)
        self.epoch = 0
        self._dataset = None
        self._identifiers = None

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _open(self):
        if self._dataset is None:
            from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2

            self._identifiers = sorted(
                nnUNetDatasetBlosc2.get_identifiers(self.data_folder)
            )
            self._dataset = nnUNetDatasetBlosc2(
                self.data_folder, identifiers=self._identifiers
            )
        return self._dataset

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        dataset = self._open()
        rng = np.random.default_rng(
            self.seed + self.epoch * 1_000_003 + int(index) * 97
        )
        identifier = self._identifiers[int(rng.integers(0, len(self._identifiers)))]
        data, segmentation, _, properties = dataset.load_case(identifier)
        _, depth, height, width = data.shape
        patch_h, patch_w = self.patch_hw
        locations = properties.get("class_locations", {}).get(1, [])
        use_foreground = (
            len(locations) > 0
            and rng.random() < self.foreground_sampling_probability
        )
        if use_foreground:
            center = locations[int(rng.integers(0, len(locations)))]
            center_z, center_y, center_x = (int(i) for i in center[-3:])
        else:
            center_z = int(rng.integers(0, max(depth, 1)))
            center_y = int(rng.integers(0, max(height, 1)))
            center_x = int(rng.integers(0, max(width, 1)))
        lower_y = max(0, min(center_y - patch_h // 2, height - patch_h))
        lower_x = max(0, min(center_x - patch_w // 2, width - patch_w))
        if height <= patch_h:
            lower_y = (height - patch_h) // 2
        if width <= patch_w:
            lower_x = (width - patch_w) // 2
        indices = [max(0, min(depth - 1, center_z + offset)) for offset in (-1, 0, 1)]
        image = data[0, indices, lower_y : lower_y + patch_h, lower_x : lower_x + patch_w]
        target = segmentation[0, center_z, lower_y : lower_y + patch_h, lower_x : lower_x + patch_w]
        if image.shape[1:] != self.patch_hw:
            image = Dataset515PatchDataset.extract(
                data[:, indices, :, :], (0, lower_y, lower_x), (3, patch_h, patch_w)
            )[0]
            target = Dataset515PatchDataset.extract(
                segmentation[:, center_z : center_z + 1, :, :],
                (0, lower_y, lower_x),
                (1, patch_h, patch_w),
            )[0, 0]
        image = np.asarray(image, dtype=np.float32)
        target = (np.asarray(target) > 0).astype(np.int64)
        for axis in (1, 2):
            if rng.random() < 0.5:
                image = np.flip(image, axis=axis).copy()
                target = np.flip(target, axis=axis).copy()
        return {"data": torch.from_numpy(image), "target": torch.from_numpy(target)}


def make_loader(
    dataset: Dataset,
    local_batch_size: int,
    rank: int,
    world_size: int,
    workers: int,
    shuffle: bool = True,
) -> tuple[DataLoader, DistributedSampler]:
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=True,
        seed=515,
    )
    return (
        DataLoader(
            dataset,
            batch_size=int(local_batch_size),
            sampler=sampler,
            num_workers=int(workers),
            pin_memory=True,
            persistent_workers=bool(workers),
            drop_last=True,
        ),
        sampler,
    )


def probe_local_batch(
    model_factory: Callable[[], torch.nn.Module],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    input_shape: Sequence[int],
    device: torch.device,
    amp_enabled: bool,
    max_batch: int,
    memory_fraction: float,
) -> int:
    """Find the largest batch below a memory ceiling using a real train step."""
    if not 0.1 <= memory_fraction < 1.0:
        raise ValueError("memory_fraction must be in [0.1, 1.0).")
    model = model_factory().to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, eps=1e-4)
    total_memory = torch.cuda.get_device_properties(device).total_memory

    def try_batch(batch_size: int) -> tuple[bool, float]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        data = target = output = loss = None
        try:
            data = torch.randn((batch_size, *input_shape), device=device)
            target = torch.zeros((batch_size, *input_shape[1:]), device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                output = model(data)
                loss = loss_fn(output, target)
            if isinstance(output, (list, tuple)):
                output = None
            if not torch.isfinite(loss):
                return False, float("inf")
            loss.backward()
            optimizer.step()
            peak = torch.cuda.max_memory_allocated(device)
            return peak <= total_memory * memory_fraction, peak / 2**30
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            return False, float("inf")
        finally:
            optimizer.zero_grad(set_to_none=True)
            data = target = output = loss = None
            torch.cuda.empty_cache()

    low = 0
    high = 1
    successful = {}
    while high <= max_batch:
        ok, peak = try_batch(high)
        successful[high] = (ok, peak)
        print(f"batch_probe batch={high} ok={ok} peak_gib={peak:.2f}", flush=True)
        if not ok:
            break
        low = high
        high *= 2
    if low == 0:
        raise RuntimeError(
            "Batch probe failed at local batch 1. Reduce the patch/model or use checkpointing."
        )
    high = min(high, max_batch)
    while low + 1 < high:
        middle = (low + high) // 2
        ok, peak = try_batch(middle)
        successful[middle] = (ok, peak)
        print(f"batch_probe batch={middle} ok={ok} peak_gib={peak:.2f}", flush=True)
        if ok:
            low = middle
        else:
            high = middle
    del model, optimizer
    torch.cuda.empty_cache()
    return int(low)


def gaussian_importance_2d(patch_hw: Sequence[int], device: torch.device) -> torch.Tensor:
    coords = [
        torch.arange(size, device=device, dtype=torch.float32) - (size - 1.0) / 2
        for size in patch_hw
    ]
    grids = torch.meshgrid(*coords, indexing="ij")
    exponent = sum((grid / max(size / 8, 1)) ** 2 for grid, size in zip(grids, patch_hw))
    return torch.exp(-0.5 * exponent).clamp_min(1e-4)


def gaussian_importance_3d(patch_size: Sequence[int], device: torch.device) -> torch.Tensor:
    coords = [
        torch.arange(size, device=device, dtype=torch.float32) - (size - 1.0) / 2
        for size in patch_size
    ]
    grids = torch.meshgrid(*coords, indexing="ij")
    exponent = sum((grid / max(size / 8, 1)) ** 2 for grid, size in zip(grids, patch_size))
    return torch.exp(-0.5 * exponent).clamp_min(1e-4)


def sliding_starts(dim: int, patch: int, step_fraction: float) -> list[int]:
    if dim <= patch:
        return [0]
    nominal_step = max(int(round(patch * step_fraction)), 1)
    count = int(math.ceil((dim - patch) / nominal_step)) + 1
    return sorted({int(round(i * (dim - patch) / (count - 1))) for i in range(count)})


def get_plan_tools():
    from batchgenerators.utilities.file_and_folder_operations import load_json
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    plans_file = PREPROCESSED_ROOT / DATASET_NAME / f"{TEST_PLAN}.json"
    if not plans_file.is_file():
        raise FileNotFoundError(f"Dataset515 test plan not found: {plans_file}")
    plans_manager = PlansManager(str(plans_file))
    configuration_manager = plans_manager.get_configuration(TEST_CONFIGURATION)
    if tuple(configuration_manager.patch_size) != PATCH_SIZE:
        raise RuntimeError(
            f"Expected test patch size {PATCH_SIZE}, got {configuration_manager.patch_size}."
        )
    dataset_json = load_json(str(PREPROCESSED_ROOT / DATASET_NAME / "dataset.json"))
    return plans_file, plans_manager, configuration_manager, dataset_json


def export_case_logits(
    logits: torch.Tensor,
    properties: dict,
    plans_manager,
    configuration_manager,
    dataset_json: dict,
    output_file: Path,
) -> None:
    from nnunetv2.inference.export_prediction import export_prediction_from_logits

    export_prediction_from_logits(
        logits.float().cpu(),
        properties,
        configuration_manager,
        plans_manager,
        dataset_json,
        str(output_file),
        False,
        1,
    )


def evaluate_predictions(
    predictions: Path,
    output_dir: Path,
    plans_file: Path,
    rank: int,
    world_size: int,
    device: torch.device,
) -> Path:
    from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder2

    barrier(world_size, device)
    if rank != 0:
        return predictions / "summary.json"
    summary = predictions / "summary.json"
    compute_metrics_on_folder2(
        str(RAW_ROOT / DATASET_NAME / "labelsTs"),
        str(predictions),
        str(PREPROCESSED_ROOT / DATASET_NAME / "dataset.json"),
        str(plans_file),
        output_file=str(summary),
        num_processes=min(16, os.cpu_count() or 1),
        chill=False,
    )
    print_summary(summary, output_dir)
    update_resultsboard(summary, output_dir)
    baseline = RESULTS_ROOT / "nnUNetTrainer__nnUNetPlans__3d_fullres" / "Test_All_Predictions_PostProcessing" / "summary.json"
    if baseline.is_file():
        subprocess.run(
            [
                str(NNUNET_PYTHON),
                str(NNUNET_ROOT / "nnunetv2" / "evaluation" / "analyze_paired_test_results.py"),
                "--candidate", str(summary),
                "--baseline", str(baseline),
                "--output-dir", str(output_dir / "paired_vs_nnunet_baseline"),
                "--label", "1",
                "--top-k", "15",
                "--bootstrap-samples", "10000",
                "--bootstrap-seed", "515",
            ],
            check=True,
        )
    return summary


def print_summary(summary_path: Path, output_dir: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = summary["foreground_mean"]
    cases = [case["metrics"]["1"] for case in summary["metric_per_case"]]
    dice = [value["Dice"] for value in cases if math.isfinite(value["Dice"])]
    print("=" * 96)
    print(f"Dataset515 independent-test result: {output_dir}")
    print("=" * 96)
    print(f"Cases       : {len(cases)}")
    print(f"Dice mean   : {metrics['Dice']:.5f}")
    print(f"Dice median : {float(np.median(dice)):.5f}")
    print(f"IoU mean    : {metrics['IoU']:.5f}")
    print(f"TP / FP / FN: {metrics['TP']:.2f} / {metrics['FP']:.2f} / {metrics['FN']:.2f}")
    print(f"Summary     : {summary_path}")
    print("=" * 96)


def update_resultsboard(summary_path: Path, output_dir: Path) -> None:
    board_path = NNUNET_ROOT / "nnUNet_results" / "resultsboard.csv"
    lock_path = NNUNET_ROOT / "nnUNet_results" / ".resultsboard.csv.lock"
    fieldnames = ["dataset", "relative_path", "fold", "Dice", "FN", "FP", "IoU", "TN", "TP", "n_pred", "n_ref"]
    metrics = json.loads(summary_path.read_text(encoding="utf-8"))["foreground_mean"]
    row = {
        "dataset": DATASET_NAME,
        "relative_path": output_dir.relative_to(RESULTS_ROOT).as_posix(),
        "fold": "Test_All_Predictions_PostProcessing",
        **{key: f"{float(metrics[key]):.5f}" for key in fieldnames[3:]},
    }
    board_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        rows = []
        if board_path.is_file() and board_path.stat().st_size:
            with board_path.open("r", encoding="utf-8", newline="") as file:
                reader = csv.DictReader(file)
                if reader.fieldnames != fieldnames:
                    raise RuntimeError(f"Unexpected resultsboard columns: {reader.fieldnames}")
                rows = list(reader)
        key = (row["dataset"], row["relative_path"], row["fold"])
        old_length = len(rows)
        rows = [value for value in rows if (value["dataset"], value["relative_path"], value["fold"]) != key]
        action = "updated" if len(rows) != old_length else "appended"
        rows.append(row)
        rows.sort(key=lambda value: (value["dataset"], value["relative_path"], value["fold"]))
        descriptor, temp_name = tempfile.mkstemp(prefix=".resultsboard.", suffix=".csv", dir=board_path.parent)
        try:
            if board_path.exists():
                os.chmod(temp_name, stat.S_IMODE(board_path.stat().st_mode))
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            os.replace(temp_name, board_path)
        except Exception:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise
    print(f"resultsboard {action}: {row['relative_path']}", flush=True)


def write_metadata(
    output_dir: Path,
    args,
    source_repository: str,
    source_commit: str,
    architecture: dict,
    world_size: int,
    local_batch_size: int,
    data_protocol: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plans_file = PREPROCESSED_ROOT / DATASET_NAME / f"{TEST_PLAN}.json"
    shutil.copy2(plans_file, output_dir / "plans.json")
    shutil.copy2(PREPROCESSED_ROOT / DATASET_NAME / "dataset.json", output_dir / "dataset.json")
    metadata = {
        "name": output_dir.name,
        "source_repository": source_repository,
        "source_commit": source_commit,
        "architecture": architecture,
        "data_protocol": data_protocol,
        "runtime": {
            "world_size": world_size,
            "local_batch_size": local_batch_size,
            "global_batch_size": local_batch_size * world_size,
            "amp": not getattr(args, "no_amp", False),
            "tf32": False,
        },
        "command_args": json_args(args),
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
