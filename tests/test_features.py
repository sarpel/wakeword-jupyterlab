#!/usr/bin/env python3
"""Unit tests for the enhanced wakeword dataset utilities."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.enhanced_dataset import EnhancedWakewordDataset, EnhancedAudioConfig, create_dataloaders


class EnhancedDatasetTestCase(unittest.TestCase):
    """Validate dataset behaviour with synthetic feature files."""

    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory()
        base = Path(self._tempdir.name)

        self.positive_dir = base / "positive_dataset"
        self.negative_dir = base / "negative_dataset"
        self.features_dir = base / "features"

        for path in (self.positive_dir, self.negative_dir, self.features_dir):
            path.mkdir(parents=True, exist_ok=True)

        (self.features_dir / "positive").mkdir(parents=True, exist_ok=True)
        (self.features_dir / "negative").mkdir(parents=True, exist_ok=True)

        np.save(self.features_dir / "positive" / "sample_pos.npy", np.random.rand(40, 8).astype(np.float32))
        np.save(self.features_dir / "negative" / "sample_neg.npy", np.random.rand(40, 8).astype(np.float32))

        self.config = EnhancedAudioConfig(
            features_dir=str(self.features_dir),
            use_precomputed_features=False,
            feature_cache_enabled=False,
        )

        self.dataset = EnhancedWakewordDataset(
            positive_dir=str(self.positive_dir),
            negative_dir=str(self.negative_dir),
            features_dir=str(self.features_dir),
            config=self.config,
            mode="train",
        )

    def tearDown(self):
        self._tempdir.cleanup()

    def test_dataset_exposes_feature_samples(self):
        self.assertEqual(len(self.dataset), 2)

        labels = {self.dataset[idx]["label"].item() for idx in range(len(self.dataset))}
        self.assertEqual(labels, {0, 1})

        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            self.assertTrue(torch.is_tensor(sample["features"]))
            self.assertEqual(sample["features"].shape, (40, 8))
            self.assertEqual(sample["source"], "feature")
            self.assertTrue(sample["path"].endswith(".npy"))

    def test_create_dataloaders_returns_batches(self):
        train_loader, val_loader = create_dataloaders(
            positive_dir=str(self.positive_dir),
            negative_dir=str(self.negative_dir),
            features_dir=str(self.features_dir),
            batch_size=1,
            config=self.config,
        )

        self.assertGreaterEqual(len(train_loader), 1)
        self.assertGreaterEqual(len(val_loader), 1)

        batch = next(iter(train_loader))
        self.assertIn("features", batch)
        self.assertEqual(batch["features"].shape[0], 1)
        self.assertIn("label", batch)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

