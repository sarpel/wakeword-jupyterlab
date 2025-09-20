#!/usr/bin/env python3
"""
Headless training runner that mirrors the Gradio app pipeline.
This allows running and supervising training from terminal without UI.
"""
import os
import sys
import time

# Ensure project root on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from gradio_app import WakewordTrainingApp


def main():
    pos = os.environ.get("POS_DIR", os.path.join(ROOT, "positive_dataset"))
    neg = os.environ.get("NEG_DIR", os.path.join(ROOT, "negative_dataset"))
    bg = os.environ.get("BG_DIR", os.path.join(ROOT, "background_noise"))
    epochs = int(os.environ.get("EPOCHS", "10"))
    batch_size = int(os.environ.get("BATCH_SIZE", "32"))
    val_split = float(os.environ.get("VAL_SPLIT", "0.2"))
    test_split = float(os.environ.get("TEST_SPLIT", "0.1"))
    lr = float(os.environ.get("LR", "1e-4"))
    dropout = float(os.environ.get("DROPOUT", "0.6"))

    app = WakewordTrainingApp()

    print("[Headless] Loading data...")
    status, train_len, val_len = app.load_data(pos, neg, bg, batch_size, val_split, test_split)
    print(status)
    if train_len is None or train_len == 0:
        print("[Headless] No training data, exiting with error.")
        sys.exit(1)

    print(f"[Headless] Train: {train_len}, Val: {val_len}")
    print("[Headless] Starting training...")
    try:
        # Run training synchronously with auto-extend and worsening/plateau detection
        best_val = app.trainer.train(
            app.train_loader,
            app.val_loader,
            epochs,
            progress_callback=None,
            auto_extend=True,
            extend_step=5,
            max_extra_epochs=20,
            plateau_delta=0.001,
            worsen_patience=3,
        )
        print(f"[Headless] Training finished. Best Val Acc: {best_val:.2f}%")
    except Exception as e:
        print(f"[Headless] Training error: {e}")
        sys.exit(2)

    # Save deployment package if best model exists
    try:
        info = app.save_model()
        print(info)
    except Exception as e:
        print(f"[Headless] Save model error: {e}")


if __name__ == "__main__":
    main()
