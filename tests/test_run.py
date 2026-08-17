from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from engine.inference import restore_directory, resolve_weights_path


class TestRunAndInference(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_dir = Path(self.temp_dir.name) / "input"
        self.output_dir = Path(self.temp_dir.name) / "output"
        self.input_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy input npy files
        for i in range(3):
            dummy_arr = np.random.uniform(0.0, 1.0, size=(128, 128)).astype(np.float32)
            np.save(self.input_dir / f"image_{i:03d}.npy", dummy_arr)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_weight_resolution(self) -> None:
        weights_path = resolve_weights_path()
        self.assertTrue(weights_path.is_file(), f"Weights path does not exist: {weights_path}")

    def test_restore_directory_execution(self) -> None:
        summary = restore_directory(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            batch_size=2,
            self_ensemble_transforms=1,  # Speed up unit test
        )

        self.assertEqual(summary["images"], 3)
        self.assertTrue(self.output_dir.exists())

        # Verify output files
        for i in range(3):
            out_file = self.output_dir / f"image_{i:03d}.npy"
            self.assertTrue(out_file.exists(), f"Missing output file: {out_file}")

            arr = np.load(out_file)

            # Check shape (256, 256)
            self.assertEqual(arr.shape, (256, 256))

            # Check dtype float32
            self.assertEqual(arr.dtype, np.float32)

            # Check range [0.0, 1.0]
            self.assertGreaterEqual(arr.min(), 0.0)
            self.assertLessEqual(arr.max(), 1.0)

            # Check no NaN or Inf values
            self.assertFalse(np.isnan(arr).any(), "NaN values found in restored output array")
            self.assertFalse(np.isinf(arr).any(), "Inf values found in restored output array")


if __name__ == "__main__":
    unittest.main()
