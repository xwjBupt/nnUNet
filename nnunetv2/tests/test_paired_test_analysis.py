import tempfile
import unittest
from pathlib import Path

import SimpleITK as sitk
import numpy as np

from nnunetv2.evaluation.analyze_paired_test_results import (
    build_analysis,
    build_case_rows,
    build_success_gate,
    statistical_inference,
)


def _metrics(n_ref: int, false_negative: int, false_positive: int) -> dict:
    true_positive = n_ref - false_negative
    dice = 2 * true_positive / (
        2 * true_positive + false_positive + false_negative
    )
    iou = true_positive / (true_positive + false_positive + false_negative)
    return {
        "Dice": dice,
        "IoU": iou,
        "TP": true_positive,
        "FP": false_positive,
        "FN": false_negative,
        "n_pred": true_positive + false_positive,
        "n_ref": n_ref,
    }


def _summary(cases: list) -> dict:
    dice = sum(case["metrics"]["1"]["Dice"] for case in cases) / len(cases)
    return {"foreground_mean": {"Dice": dice}, "metric_per_case": cases}


class TestPairedTestAnalysis(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        volumes_voxels = (500, 2000, 10000, 25000)
        candidate_cases = []
        baseline_cases = []
        for index, n_ref in enumerate(volumes_voxels):
            reference = root / f"case_{index}.nii.gz"
            candidate_prediction = root / "candidate" / reference.name
            baseline_prediction = root / "baseline" / reference.name
            reference_array = np.zeros((32, 32, 32), dtype=np.uint8)
            reference_array.flat[:n_ref] = 1
            baseline_array = np.zeros_like(reference_array)
            baseline_array.flat[: n_ref - 100] = 1
            baseline_array.flat[n_ref : n_ref + 100] = 1
            candidate_array = np.zeros_like(reference_array)
            candidate_array.flat[: n_ref - 90] = 1
            candidate_array.flat[n_ref : n_ref + 95] = 1

            image = sitk.GetImageFromArray(reference_array)
            baseline_image = sitk.GetImageFromArray(baseline_array)
            candidate_image = sitk.GetImageFromArray(candidate_array)
            for item in (image, baseline_image, candidate_image):
                item.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(image, str(reference))
            candidate_prediction.parent.mkdir(exist_ok=True)
            baseline_prediction.parent.mkdir(exist_ok=True)
            sitk.WriteImage(candidate_image, str(candidate_prediction))
            sitk.WriteImage(baseline_image, str(baseline_prediction))
            baseline_cases.append(
                {
                    "reference_file": str(reference),
                    "prediction_file": str(baseline_prediction),
                    "metrics": {"1": _metrics(n_ref, 100, 100)},
                }
            )
            candidate_cases.append(
                {
                    "reference_file": str(reference),
                    "prediction_file": str(candidate_prediction),
                    "metrics": {"1": _metrics(n_ref, 90, 95)},
                }
            )
        self.baseline = _summary(baseline_cases)
        self.candidate = _summary(candidate_cases)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_volume_strata_physical_errors_and_strict_gate(self):
        rows = build_case_rows(self.candidate, self.baseline, "1")
        analysis = build_analysis(rows, top_k=2)
        gate = build_success_gate(analysis, self.candidate, self.baseline)

        self.assertEqual(
            [row["volume_group"] for row in rows],
            ["lt_1ml", "1_to_5ml", "5_to_20ml", "ge_20ml"],
        )
        self.assertEqual(
            [analysis["groups"][name]["case_count"] for name in (
                "lt_1ml", "1_to_5ml", "5_to_20ml", "ge_20ml"
            )],
            [1, 1, 1, 1],
        )
        self.assertAlmostEqual(rows[0]["reference_volume_ml"], 0.5)
        self.assertAlmostEqual(rows[0]["delta_FN_ml"], -0.01)
        self.assertAlmostEqual(rows[0]["delta_FP_ml"], -0.005)
        all_cases = analysis["groups"]["all"]["paired_delta"]
        self.assertAlmostEqual(all_cases["FN_ml_mean"], -0.01)
        self.assertAlmostEqual(all_cases["FP_ml_mean"], -0.005)
        candidate = analysis["groups"]["all"]["candidate"]
        self.assertEqual(candidate["zero_overlap_cases"], 0)
        self.assertEqual(candidate["severe_failure_cases_dice_lt_0_2"], 0)
        self.assertGreater(candidate["Dice_p10"], 0.2)
        self.assertIn("delta_FN_ml", analysis["largest_dice_losses"][0])
        self.assertIn("delta_FP_ml", analysis["largest_dice_gains"][0])
        self.assertTrue(gate["passed"])
        self.assertGreater(gate["delta"], 0)

    def test_equal_result_does_not_pass_and_statistics_are_defined(self):
        rows = build_case_rows(self.baseline, self.baseline, "1")
        analysis = build_analysis(rows, top_k=2)
        gate = build_success_gate(analysis, self.baseline, self.baseline)
        inference = statistical_inference(rows, bootstrap_samples=100, bootstrap_seed=515)

        self.assertFalse(gate["passed"])
        self.assertEqual(gate["delta"], 0)
        paired = analysis["groups"]["all"]["paired_delta"]
        self.assertEqual((paired["wins"], paired["ties"], paired["losses"]), (0, 4, 0))
        bootstrap = inference["paired_dice_delta_bootstrap"]
        self.assertEqual(bootstrap["confidence_interval_percentile"], [0.0, 0.0])
        self.assertEqual(
            inference["paired_dice_delta_wilcoxon"]["p_value_two_sided"], 1.0
        )
        self.assertIsNone(
            inference["reference_volume_vs_dice_delta_spearman"]["rho"]
        )

    def test_case_set_mismatch_is_rejected(self):
        incomplete_candidate = _summary(self.candidate["metric_per_case"][:-1])
        with self.assertRaisesRegex(RuntimeError, "case sets differ"):
            build_case_rows(incomplete_candidate, self.baseline, "1")

    def test_missing_prediction_is_rejected(self):
        prediction = Path(
            self.candidate["metric_per_case"][0]["prediction_file"]
        )
        prediction.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "prediction does not exist"):
            build_case_rows(self.candidate, self.baseline, "1")

    def test_summary_metric_mismatch_is_rejected(self):
        self.candidate["metric_per_case"][0]["metrics"]["1"]["FN"] += 1
        with self.assertRaisesRegex(RuntimeError, "summary metric differs"):
            build_case_rows(self.candidate, self.baseline, "1")


if __name__ == "__main__":
    unittest.main()
