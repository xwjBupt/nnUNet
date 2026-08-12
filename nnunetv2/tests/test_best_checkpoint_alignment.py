import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSegMambaUI import (
    nnUNetTrainerSegMambaUI,
)


class TestBestCheckpointAlignment(unittest.TestCase):
    def make_trainer(self, root: Path, is_ddp: bool):
        trainer = object.__new__(nnUNetTrainerSegMambaUI)
        trainer.output_folder = str(root)
        trainer.best_val_checkpoint_file = str(root / "checkpoint_best_val.pth")
        trainer.is_ddp = is_ddp
        trainer.device = torch.device("cpu")
        trainer.load_checkpoint = Mock()
        trainer.print_to_log_file = Mock()
        return trainer

    def test_single_process_copies_and_loads_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            trainer = self.make_trainer(root, is_ddp=False)
            best = root / "checkpoint_best_val.pth"
            final = root / "checkpoint_final.pth"
            best.write_bytes(b"best weights")
            final.write_bytes(b"last epoch weights")

            trainer._synchronize_best_checkpoint_for_final_validation()

            self.assertEqual(final.read_bytes(), b"best weights")
            trainer.load_checkpoint.assert_called_once_with(str(final))

    def test_missing_best_checkpoint_keeps_last_epoch_final(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            trainer = self.make_trainer(root, is_ddp=False)
            final = root / "checkpoint_final.pth"
            final.write_bytes(b"last epoch weights")

            trainer._synchronize_best_checkpoint_for_final_validation()

            self.assertEqual(final.read_bytes(), b"last epoch weights")
            trainer.load_checkpoint.assert_not_called()

    def test_non_main_rank_loads_final_after_success_broadcast(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            trainer = self.make_trainer(root, is_ddp=True)
            final = root / "checkpoint_final.pth"
            final.write_bytes(b"best weights copied by rank zero")

            def broadcast_success(value, src):
                self.assertEqual(src, 0)
                value.fill_(1)

            with (
                patch(
                    "nnunetv2.training.nnUNetTrainer."
                    "nnUNetTrainerSegMambaUI.dist.get_rank",
                    return_value=1,
                ),
                patch(
                    "nnunetv2.training.nnUNetTrainer."
                    "nnUNetTrainerSegMambaUI.dist.broadcast",
                    side_effect=broadcast_success,
                ),
            ):
                trainer._synchronize_best_checkpoint_for_final_validation()

            trainer.load_checkpoint.assert_called_once_with(str(final))


if __name__ == "__main__":
    unittest.main()
