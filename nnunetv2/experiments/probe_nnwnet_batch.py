#!/usr/bin/env python3
"""Select a safe WNet3D local batch size with a real AMP training step."""

from __future__ import annotations

import argparse
import gc
import json
import os

import torch
import torch.nn.functional as F

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_WNet3D import WNet3D


PATCH_SIZE = (128, 96, 96)


def parse_int_list(value: str, expected_length: int, name: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers") from error
    if len(parsed) != expected_length:
        raise argparse.ArgumentTypeError(f"{name} requires {expected_length} integers")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-local-batch", type=int, default=8)
    parser.add_argument("--memory-fraction", type=float, default=0.85)
    parser.add_argument("--layer-channels", default="32,64,128,256,320")
    parser.add_argument("--global-dims", default="16,32,64,128,160")
    parser.add_argument("--num-heads", default="1,2,4,8")
    parser.add_argument("--sr-ratio", default="8,4,2,1")
    parser.add_argument("--init-std", type=float, default=1e-2)
    return parser.parse_args()


def deep_supervision_loss(outputs, target: torch.Tensor) -> torch.Tensor:
    weights = [1.0 / (2**index) for index in range(len(outputs))]
    weights[-1] = 0.0
    weight_sum = sum(weights)
    total = target.new_zeros((), dtype=torch.float32)
    for weight, logits in zip(weights, outputs):
        if weight == 0.0:
            continue
        downsampled_target = F.interpolate(
            target.unsqueeze(1).float(), size=logits.shape[2:], mode="nearest"
        ).squeeze(1).long()
        cross_entropy = F.cross_entropy(logits, downsampled_target)
        probability = torch.softmax(logits, dim=1)[:, 1]
        reference = (downsampled_target == 1).float()
        axes = tuple(range(1, probability.ndim))
        dice = 1.0 - (
            (2.0 * (probability * reference).sum(axes) + 1e-5)
            / (probability.sum(axes) + reference.sum(axes) + 1e-5)
        ).mean()
        total = total + weight * (cross_entropy + dice) / weight_sum
    return total


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for nnWNet batch probing")
    if args.max_local_batch < 1:
        raise ValueError("--max-local-batch must be positive")
    if not 0.5 <= args.memory_fraction < 1.0:
        raise ValueError("--memory-fraction must be in [0.5, 1.0)")

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    layer_channels = parse_int_list(args.layer_channels, 5, "--layer-channels")
    global_dims = parse_int_list(args.global_dims, 5, "--global-dims")
    num_heads = parse_int_list(args.num_heads, 4, "--num-heads")
    sr_ratio = parse_int_list(args.sr_ratio, 4, "--sr-ratio")
    model = WNet3D(
        in_channel=1,
        num_classes=2,
        deep_supervised=True,
        layer_channel=layer_channels,
        global_dim=global_dims,
        num_heads=num_heads,
        sr_ratio=sr_ratio,
    ).to(device)
    model.train()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=1e-2, momentum=0.99, nesterov=True, weight_decay=3e-5
    )
    scaler = torch.amp.GradScaler("cuda")
    memory_limit = (
        torch.cuda.get_device_properties(device).total_memory * args.memory_fraction
    )

    def attempt(batch_size: int) -> tuple[bool, float]:
        data = target = output = loss = None
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            data = torch.randn((batch_size, 1, *PATCH_SIZE), device=device)
            target = torch.zeros((batch_size, *PATCH_SIZE), device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(data)
                loss = deep_supervision_loss(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
            return peak <= memory_limit, peak / 2**30
        except torch.OutOfMemoryError:
            return False, float("inf")
        finally:
            optimizer.zero_grad(set_to_none=True)
            data = target = output = loss = None
            torch.cuda.empty_cache()
            gc.collect()

    best = 0
    for batch_size in range(1, args.max_local_batch + 1):
        ok, peak_gib = attempt(batch_size)
        print(
            f"batch_probe local_batch={batch_size} ok={ok} peak_gib={peak_gib:.2f}",
            flush=True,
        )
        if not ok:
            break
        best = batch_size
    if best == 0:
        raise RuntimeError("nnWNet cannot complete a local batch of one")
    print(
        json.dumps(
            {
                "local_batch_size": best,
                "memory_fraction": args.memory_fraction,
                "visible_device": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
