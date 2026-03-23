"""Train the trajectory event classifier on noisy synthetic data.

Usage:
    python -m tools.train_event_classifier
    python -m tools.train_event_classifier --epochs 50 --device cpu
"""

import argparse
import glob
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from app.pipeline.event_classifier import train_model, evaluate, EventClassifierNet, TrajectoryEventDataset
from torch.utils.data import DataLoader
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/synth/noisy/")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save", default="model_weight/event_classifier.pt")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction of files for validation")
    args = parser.parse_args()

    # Find all noisy variant files
    files = sorted(glob.glob(f"{args.data_dir}/noisy_*.json"))
    if not files:
        print(f"No training data found in {args.data_dir}")
        sys.exit(1)

    # Split into train/val
    n_val = max(1, int(len(files) * args.val_split))
    val_files = files[-n_val:]
    train_files = files[:-n_val]

    logging.info("Train files: %d, Val files: %d", len(train_files), len(val_files))

    model = train_model(
        train_paths=train_files,
        val_paths=val_files,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        save_path=args.save,
    )

    # Final evaluation on val set
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    val_ds = TrajectoryEventDataset(val_files, oversample_events=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    metrics = evaluate(model, val_dl, device)

    print("\n=== Final Validation Results ===")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Macro F1 (events): {metrics['macro_f1']:.3f}")
    for name in ["fly", "bounce", "hit", "serve"]:
        p = metrics.get(f"{name}_precision", 0)
        r = metrics.get(f"{name}_recall", 0)
        f1 = metrics.get(f"{name}_f1", 0)
        print(f"  {name:8s}: P={p:.3f} R={r:.3f} F1={f1:.3f}")


if __name__ == "__main__":
    main()
