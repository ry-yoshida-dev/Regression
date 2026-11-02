from __future__ import annotations

import unittest

import numpy as np

from regression import (
    GroupKFoldSplitter,
    HoldoutSplitter,
    KFoldSplitter,
    SupervisedDataset,
    TimeSeriesSplitter,
)
from tests.helpers import linear_data


class TestHoldoutSplitter(unittest.TestCase):
    def test_partition_sizes(self) -> None:
        X, y = linear_data(num_samples=100)
        splits = HoldoutSplitter(test_fraction=0.25).split(SupervisedDataset(X, y))
        self.assertEqual(len(splits), 1)
        self.assertEqual(splits[0].test_indices.size, 25)
        self.assertEqual(splits[0].train_indices.size, 75)

    def test_invalid_fraction_raises(self) -> None:
        for test_fraction in (0.0, 1.0):
            with self.subTest(test_fraction=test_fraction):
                with self.assertRaises(ValueError):
                    _ = HoldoutSplitter(test_fraction=test_fraction)


class TestKFoldSplitter(unittest.TestCase):
    def test_test_partitions_cover_all_samples_once(self) -> None:
        X, y = linear_data()
        splits = KFoldSplitter(n_splits=3).split(SupervisedDataset(X, y))
        test_indices = np.concatenate([split.test_indices for split in splits])
        self.assertEqual(np.sort(test_indices).tolist(), list(range(y.size)))
        for split in splits:
            self.assertEqual(
                np.intersect1d(split.train_indices, split.test_indices).size, 0
            )

    def test_is_reproducible(self) -> None:
        X, y = linear_data()
        dataset = SupervisedDataset(X, y)
        first = KFoldSplitter(random_state=1).split(dataset)
        second = KFoldSplitter(random_state=1).split(dataset)
        np.testing.assert_array_equal(first[0].test_indices, second[0].test_indices)

    def test_unshuffled_ignores_random_state(self) -> None:
        X, y = linear_data()
        splits = KFoldSplitter(is_shuffled=False, random_state=1).split(
            SupervisedDataset(X, y)
        )
        self.assertEqual(len(splits), 5)

    def test_invalid_n_splits_raises(self) -> None:
        with self.assertRaises(ValueError):
            _ = KFoldSplitter(n_splits=1)


class TestGroupKFoldSplitter(unittest.TestCase):
    def test_groups_never_cross_partitions(self) -> None:
        X, y = linear_data()
        groups = np.arange(y.size) % 12
        splits = GroupKFoldSplitter(groups=groups, n_splits=3).split(
            SupervisedDataset(X, y)
        )
        for split in splits:
            self.assertEqual(
                np.intersect1d(
                    groups[split.train_indices], groups[split.test_indices]
                ).size,
                0,
            )

    def test_group_length_mismatch_raises(self) -> None:
        X, y = linear_data()
        splitter = GroupKFoldSplitter(groups=np.array([0, 1]))
        with self.assertRaises(ValueError):
            _ = splitter.split(SupervisedDataset(X, y))


class TestTimeSeriesSplitter(unittest.TestCase):
    def test_training_precedes_test_with_gap(self) -> None:
        X, y = linear_data()
        splits = TimeSeriesSplitter(n_splits=3, gap=2).split(SupervisedDataset(X, y))
        for split in splits:
            self.assertEqual(
                int(split.train_indices.max()) + 3, int(split.test_indices.min())
            )

    def test_invalid_arguments_raise(self) -> None:
        with self.assertRaises(ValueError):
            _ = TimeSeriesSplitter(n_splits=1)
        with self.assertRaises(ValueError):
            _ = TimeSeriesSplitter(gap=-1)
        with self.assertRaises(ValueError):
            _ = TimeSeriesSplitter(test_size=0)


if __name__ == "__main__":
    _ = unittest.main()
