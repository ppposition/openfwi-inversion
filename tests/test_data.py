import tempfile
import unittest
from pathlib import Path

import numpy as np

from openfwi_inversion.data import split_velocity_batch


class SplitVelocityBatchTests(unittest.TestCase):
    def test_splits_batch_into_individual_models(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = np.arange(5 * 1 * 2 * 3, dtype=np.float32).reshape(5, 1, 2, 3)
            source_path = root / "model1.npy"
            np.save(source_path, source)

            outputs = split_velocity_batch(source_path, root / "split")

            self.assertEqual([path.name for path in outputs], [f"model1_{i:05d}.npy" for i in range(5)])
            self.assertEqual(np.load(outputs[0]).shape, (1, 2, 3))
            np.testing.assert_array_equal(np.load(outputs[3]), source[3])

    def test_splits_batch_into_smaller_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = np.arange(5 * 1 * 2 * 3).reshape(5, 1, 2, 3)
            source_path = root / "models.npy"
            np.save(source_path, source)

            outputs = split_velocity_batch(source_path, root / "split", samples_per_file=2)

            self.assertEqual(len(outputs), 3)
            self.assertEqual(np.load(outputs[0]).shape, (2, 1, 2, 3))
            self.assertEqual(np.load(outputs[-1]).shape, (1, 1, 2, 3))

    def test_rejects_wrong_shape_and_existing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            wrong_path = root / "wrong.npy"
            np.save(wrong_path, np.zeros((5, 2, 3)))
            with self.assertRaisesRegex(ValueError, "N, 1, H, W"):
                split_velocity_batch(wrong_path, root / "split")

            source_path = root / "models.npy"
            np.save(source_path, np.zeros((2, 1, 2, 3)))
            split_velocity_batch(source_path, root / "split")
            with self.assertRaises(FileExistsError):
                split_velocity_batch(source_path, root / "split")


if __name__ == "__main__":
    unittest.main()
