from __future__ import annotations

import unittest

import numpy as np

from regression import SupervisedDataset


class TestSupervisedDataset(unittest.TestCase):
    def test_valid_dataset(self) -> None:
        dataset = SupervisedDataset(np.zeros((3, 2)), np.array([1.0, 2.0, 3.0]))
        self.assertEqual(dataset.num_samples, 3)

    def test_integer_targets_are_accepted(self) -> None:
        dataset = SupervisedDataset(np.zeros((2, 1)), np.array([1, 2]))
        self.assertEqual(dataset.y.tolist(), [1, 2])

    def test_subset(self) -> None:
        dataset = SupervisedDataset(
            np.arange(6.0).reshape(3, 2), np.array([0.0, 1.0, 2.0])
        )
        subset = dataset.subset(np.array([2, 0], dtype=np.intp))
        self.assertEqual(subset.y.tolist(), [2.0, 0.0])
        np.testing.assert_array_equal(subset.X, [[4.0, 5.0], [0.0, 1.0]])

    def test_invalid_feature_shape_raises(self) -> None:
        with self.assertRaises(ValueError):
            _ = SupervisedDataset(np.zeros(3), np.zeros(3))
        with self.assertRaises(ValueError):
            _ = SupervisedDataset(np.zeros((0, 2)), np.zeros(0))

    def test_non_numeric_inputs_raise(self) -> None:
        with self.assertRaises(TypeError):
            _ = SupervisedDataset(np.array([["a"]]), np.array([0.0]))
        with self.assertRaises(TypeError):
            _ = SupervisedDataset(np.zeros((1, 1)), np.array(["a"]))

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            _ = SupervisedDataset(np.zeros((2, 1)), np.zeros(3))

    def test_multi_output_targets_raise(self) -> None:
        with self.assertRaises(ValueError):
            _ = SupervisedDataset(np.zeros((2, 1)), np.zeros((2, 1)))

    def test_non_finite_targets_raise(self) -> None:
        for value in (np.nan, np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _ = SupervisedDataset(np.zeros((2, 1)), np.array([0.0, value]))


if __name__ == "__main__":
    _ = unittest.main()
