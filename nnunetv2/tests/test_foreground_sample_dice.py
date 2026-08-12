import os
import socket
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nnunetv2.training.loss.dice import ForegroundSampleDiceLoss


def _make_uneven_batch(rank: int, parameter: torch.Tensor):
    target = torch.zeros(2, 1, 1, 1, 8, dtype=torch.long)
    if rank == 0:
        target[1, 0, 0, 0, :1] = 1
    else:
        target[0, 0, 0, 0, :5] = 1
        target[1, 0, 0, 0, :8] = 1
    offsets = torch.tensor(
        [[-0.6, 0.3], [0.1, -0.2]]
        if rank == 0
        else [[0.5, -0.4], [-0.1, 0.7]],
        dtype=torch.float32,
    ).view(2, 2, 1, 1, 1)
    weights = torch.tensor([[-0.7, 1.2]], dtype=torch.float32).view(
        1, 2, 1, 1, 1
    )
    logits = (parameter * weights + offsets).expand(-1, -1, 1, 1, 8)
    return logits, target


def _ddp_equivalence_worker(rank: int, world_size: int, port: int, result_queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        parameter = torch.tensor(0.35, requires_grad=True)
        logits, target = _make_uneven_batch(rank, parameter)
        loss = ForegroundSampleDiceLoss(
            apply_nonlin=lambda value: torch.softmax(value, dim=1),
            batch_dice=False,
            do_bg=False,
            smooth=1e-5,
            ddp=True,
        )(logits, target)
        loss.backward()

        rank_losses = [torch.zeros_like(loss) for _ in range(world_size)]
        dist.all_gather(rank_losses, loss.detach())
        averaged_gradient = parameter.grad.detach().clone()
        dist.all_reduce(averaged_gradient)
        averaged_gradient /= world_size

        if rank == 0:
            reference_parameter = torch.tensor(0.35, requires_grad=True)
            batches = [
                _make_uneven_batch(item_rank, reference_parameter)
                for item_rank in range(world_size)
            ]
            reference_loss = ForegroundSampleDiceLoss(
                apply_nonlin=lambda value: torch.softmax(value, dim=1),
                batch_dice=False,
                do_bg=False,
                smooth=1e-5,
                ddp=False,
            )(
                torch.cat([item[0] for item in batches], dim=0),
                torch.cat([item[1] for item in batches], dim=0),
            )
            reference_loss.backward()
            result_queue.put(
                (
                    [float(item) for item in rank_losses],
                    float(reference_loss),
                    float(averaged_gradient),
                    float(reference_parameter.grad),
                )
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


class TestForegroundSampleDiceLoss(unittest.TestCase):
    def test_matches_manual_foreground_sample_mean(self):
        torch.manual_seed(515)
        logits = torch.randn(4, 2, 1, 1, 8, requires_grad=True)
        target = torch.zeros(4, 1, 1, 1, 8, dtype=torch.long)
        target[1, 0, 0, 0, :1] = 1
        target[2, 0, 0, 0, :5] = 1
        target[3, 0, 0, 0, :8] = 1

        loss = ForegroundSampleDiceLoss(
            apply_nonlin=lambda value: torch.softmax(value, dim=1),
            batch_dice=False,
            do_bg=False,
            smooth=1e-5,
            ddp=False,
        )(logits, target)

        probability = torch.softmax(logits, dim=1)[:, 1:]
        foreground = (target == 1).float()
        intersection = (probability * foreground).sum((2, 3, 4))
        sum_prediction = probability.sum((2, 3, 4))
        sum_target = foreground.sum((2, 3, 4))
        dice = (2 * intersection + 1e-5) / (
            sum_prediction + sum_target + 1e-5
        )
        expected = -dice[sum_target > 0].mean()

        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(torch.count_nonzero(logits.grad[0]).item(), 0)
        self.assertGreater(torch.count_nonzero(logits.grad[1:]).item(), 0)

    def test_all_empty_targets_return_differentiable_zero(self):
        logits = torch.randn(2, 2, 4, 4, 4, requires_grad=True)
        target = torch.zeros(2, 1, 4, 4, 4, dtype=torch.long)
        loss = ForegroundSampleDiceLoss(
            apply_nonlin=lambda value: torch.softmax(value, dim=1),
            batch_dice=False,
            do_bg=False,
            smooth=1e-5,
            ddp=False,
        )(logits, target)
        self.assertTrue(loss.requires_grad)
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertEqual(torch.count_nonzero(logits.grad).item(), 0)

    def test_two_rank_value_and_gradient_match_global_batch(self):
        context = mp.get_context("spawn")
        result_queue = context.SimpleQueue()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        mp.start_processes(
            _ddp_equivalence_worker,
            args=(2, port, result_queue),
            nprocs=2,
            join=True,
            start_method="spawn",
        )
        rank_losses, reference_loss, averaged_gradient, reference_gradient = (
            result_queue.get()
        )
        for rank_loss in rank_losses:
            self.assertAlmostEqual(rank_loss, reference_loss, places=6)
        self.assertAlmostEqual(averaged_gradient, reference_gradient, places=6)


if __name__ == "__main__":
    unittest.main()
