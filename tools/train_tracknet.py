"""Fine-tune TrackNet for tennis ball detection.

Trains TrackNet from pre-trained weights using in-house + public data,
with progressive unfreezing and weighted focal loss.

Usage:
    python -m tools.train_tracknet --data data/tracknet_training --epochs 20
    python -m tools.train_tracknet --data data/tracknet_training --pretrained model_weight/TrackNet_best.pt
    python -m tools.train_tracknet --resume model_weight/tracknet_finetuned/last.pt
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

TRACKNET_H, TRACKNET_W = 288, 512
SEQ_LEN = 8
GAUSSIAN_SIGMA = 2.5


# ── Dataset ───────────────────────────────────────────────────────────────────


def generate_gaussian_heatmap(
    x: float, y: float, h: int = TRACKNET_H, w: int = TRACKNET_W, sigma: float = GAUSSIAN_SIGMA,
) -> np.ndarray:
    """Generate a 2D Gaussian heatmap centered at (x, y)."""
    heatmap = np.zeros((h, w), dtype=np.float32)
    if x < 0 or y < 0 or x >= w or y >= h:
        return heatmap

    # Create coordinate grids around the center
    radius = int(3 * sigma + 1)
    x_int, y_int = int(round(x)), int(round(y))

    y_min = max(0, y_int - radius)
    y_max = min(h, y_int + radius + 1)
    x_min = max(0, x_int - radius)
    x_max = min(w, x_int + radius + 1)

    yy, xx = np.meshgrid(
        np.arange(y_min, y_max, dtype=np.float32),
        np.arange(x_min, x_max, dtype=np.float32),
        indexing="ij",
    )

    gaussian = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    heatmap[y_min:y_max, x_min:x_max] = gaussian

    return heatmap


def compute_median_bg(frame_dir: Path, input_size: tuple[int, int], max_samples: int = 200) -> np.ndarray:
    """Compute background median from frame directory.

    Returns (3, H, W) float32 in [0, 1], matching TrackNetDetector preprocessing.
    """
    all_frames = sorted(frame_dir.glob("*.jpg"))
    if not all_frames:
        all_frames = sorted(frame_dir.glob("*.png"))
    if not all_frames:
        return np.zeros((3, input_size[0], input_size[1]), dtype=np.float32)

    n_samples = min(max_samples, len(all_frames))
    step = max(1, len(all_frames) // n_samples)
    sampled = all_frames[::step][:n_samples]

    h, w = input_size
    frames = []
    for fp in sampled:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        img = cv2.resize(img, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img.astype(np.float32) / 255.0)

    if not frames:
        return np.zeros((3, h, w), dtype=np.float32)

    median = np.median(np.stack(frames), axis=0)
    return median.transpose(2, 0, 1).astype(np.float32)  # CHW


class TrackNetDataset(Dataset):
    """Dataset for TrackNet training.

    Each sample = seq_len consecutive frames + background median → heatmap targets.

    Input:  [bg(3ch), f1(3ch), ..., f_seq(3ch)] = (seq_len+1)*3 channels at 288×512
    Target: [hm1, hm2, ..., hm_seq] = seq_len heatmaps at 288×512

    Preprocessing matches TrackNetDetector exactly:
        BGR → RGB, resize to 288×512, /255.0 (no ImageNet normalization)
    """

    def __init__(
        self,
        match_dirs: list[Path],
        seq_len: int = SEQ_LEN,
        input_size: tuple[int, int] = (TRACKNET_H, TRACKNET_W),
        augment: bool = True,
        sigma: float = GAUSSIAN_SIGMA,
    ):
        self.seq_len = seq_len
        self.input_h, self.input_w = input_size
        self.augment = augment
        self.sigma = sigma

        # Load all matches and build sample index
        self.samples = []  # List of (match_idx, start_frame)
        self.matches = []  # List of match data dicts

        for match_dir in match_dirs:
            match_data = self._load_match(match_dir)
            if match_data is None:
                continue

            match_idx = len(self.matches)
            self.matches.append(match_data)

            # Generate valid sequences from contiguous frame runs
            for run_start, run_end in match_data["contiguous_runs"]:
                for start in range(run_start, run_end - seq_len + 1):
                    self.samples.append((match_idx, start))

        log.info("TrackNetDataset: %d matches, %d samples", len(self.matches), len(self.samples))

    def _load_match(self, match_dir: Path) -> dict | None:
        """Load a single match: frame paths + CSV labels + background median."""
        frame_dir = match_dir / "frame"
        csv_dir = match_dir / "csv"

        if not frame_dir.exists():
            return None

        # Find CSV
        csv_files = list(csv_dir.glob("*.csv")) if csv_dir.exists() else []
        # Also check for labels.csv in match root
        root_csv = match_dir / "labels.csv"
        if root_csv.exists():
            csv_files.append(root_csv)

        if not csv_files:
            log.warning("No CSV found for %s", match_dir)
            return None

        # Detect source resolution from a sample frame to compute coordinate scaling.
        # In-house data (match_cam*): coords already scaled to training res by prepare script
        # Public data (public_*): coords at original frame resolution, need scaling
        scale_x, scale_y = 1.0, 1.0
        is_public = match_dir.name.startswith("public_")
        if is_public:
            sample_frames = sorted(frame_dir.glob("*.jpg"))[:1] or sorted(frame_dir.glob("*.png"))[:1]
            if sample_frames:
                sample_img = cv2.imread(str(sample_frames[0]))
                if sample_img is not None:
                    src_h, src_w = sample_img.shape[:2]
                    if src_w != self.input_w or src_h != self.input_h:
                        scale_x = self.input_w / src_w
                        scale_y = self.input_h / src_h

        # Parse labels
        labels = {}  # {frame_idx: (visibility, x, y)}
        for csv_path in csv_files:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        fi = int(row["Frame"])
                        vis = int(float(row["Visibility"]))
                        x = float(row["X"]) * scale_x
                        y = float(row["Y"]) * scale_y
                        labels[fi] = (vis, x, y)
                    except (ValueError, KeyError):
                        continue

        # Find available frames
        available_frames = set()
        for fp in frame_dir.iterdir():
            if fp.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                try:
                    available_frames.add(int(fp.stem))
                except ValueError:
                    continue

        # Build contiguous runs (sequences of frames with gap <= 1)
        sorted_frames = sorted(available_frames & set(labels.keys()))
        contiguous_runs = []
        if sorted_frames:
            run_start = sorted_frames[0]
            prev = sorted_frames[0]
            for fi in sorted_frames[1:]:
                if fi - prev > 2:  # Allow gap of 1 frame
                    if prev - run_start + 1 >= self.seq_len:
                        contiguous_runs.append((run_start, prev + 1))
                    run_start = fi
                prev = fi
            if prev - run_start + 1 >= self.seq_len:
                contiguous_runs.append((run_start, prev + 1))

        # Compute background median
        bg = compute_median_bg(frame_dir, (self.input_h, self.input_w))

        return {
            "frame_dir": frame_dir,
            "labels": labels,
            "bg": bg,
            "contiguous_runs": contiguous_runs,
            "name": match_dir.name,
        }

    def _load_frame(self, frame_dir: Path, fi: int) -> np.ndarray:
        """Load and preprocess a single frame → (3, H, W) float32 [0, 1]."""
        for ext in [".jpg", ".jpeg", ".png"]:
            fp = frame_dir / f"{fi}{ext}"
            if fp.exists():
                img = cv2.imread(str(fp))
                if img is not None:
                    img = cv2.resize(img, (self.input_w, self.input_h))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    return (img.astype(np.float32) / 255.0).transpose(2, 0, 1)

        # Fallback: zero frame
        return np.zeros((3, self.input_h, self.input_w), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        match_idx, start_frame = self.samples[idx]
        match = self.matches[match_idx]

        # Load sequence of frames
        frames = []
        heatmaps = []

        for i in range(self.seq_len):
            fi = start_frame + i
            frame = self._load_frame(match["frame_dir"], fi)
            frames.append(frame)

            # Generate heatmap target
            label = match["labels"].get(fi, (0, 0, 0))
            vis, x, y = label

            if vis > 0 and x > 0 and y > 0:
                hm = generate_gaussian_heatmap(x, y, self.input_h, self.input_w, self.sigma)
            else:
                hm = np.zeros((self.input_h, self.input_w), dtype=np.float32)

            heatmaps.append(hm)

        # Stack: [bg, f1, f2, ..., f8] → (27, H, W)
        bg = match["bg"].copy()
        input_frames = np.concatenate([bg[np.newaxis]] + [f[np.newaxis] for f in frames], axis=0)
        input_tensor = input_frames.reshape(-1, self.input_h, self.input_w)  # (27, H, W)

        # Target: (8, H, W)
        target_tensor = np.stack(heatmaps, axis=0)

        # Augmentation
        if self.augment:
            input_tensor, target_tensor = self._augment(input_tensor, target_tensor, match["labels"],
                                                        start_frame)

        return torch.from_numpy(input_tensor), torch.from_numpy(target_tensor)

    def _augment(
        self,
        input_tensor: np.ndarray,
        target_tensor: np.ndarray,
        labels: dict,
        start_frame: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation to input and target tensors."""
        # Horizontal flip (50%)
        if np.random.random() < 0.5:
            input_tensor = input_tensor[:, :, ::-1].copy()
            target_tensor = target_tensor[:, :, ::-1].copy()

        # Temporal reverse (50%)
        if np.random.random() < 0.5:
            # Reverse frame order (keep bg at front)
            n_ch = 3  # channels per frame
            bg = input_tensor[:n_ch]  # (3, H, W)
            seq_frames = input_tensor[n_ch:]  # (24, H, W)
            # Reshape to (8, 3, H, W), reverse, reshape back
            seq_reshaped = seq_frames.reshape(self.seq_len, n_ch, self.input_h, self.input_w)
            seq_reversed = seq_reshaped[::-1].copy()
            input_tensor = np.concatenate([bg[np.newaxis], seq_reversed], axis=0).reshape(
                -1, self.input_h, self.input_w)
            target_tensor = target_tensor[::-1].copy()

        # Brightness/contrast jitter (30%)
        if np.random.random() < 0.3:
            brightness = np.random.uniform(0.8, 1.2)
            contrast = np.random.uniform(0.8, 1.2)
            input_tensor = np.clip(input_tensor * contrast + (brightness - 1.0), 0.0, 1.0).astype(
                np.float32)

        return input_tensor, target_tensor


# ── Loss Function ─────────────────────────────────────────────────────────────


class WeightedFocalLoss(nn.Module):
    """Focal loss with positive class weighting for extreme imbalance.

    In TrackNet heatmaps, positive pixels (ball location) are < 0.01% of total.
    pos_weight upweights the rare positive pixels.
    focal gamma suppresses easy negatives.
    """

    def __init__(self, gamma: float = 2.0, pos_weight: float = 50.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute focal loss between predicted and target heatmaps.

        Args:
            pred: (B, C, H, W) predicted heatmaps after sigmoid
            target: (B, C, H, W) ground truth heatmaps
        """
        eps = 1e-7
        pred = pred.clamp(eps, 1.0 - eps)

        # Binary cross entropy with pos_weight
        bce = -(self.pos_weight * target * torch.log(pred) + (1 - target) * torch.log(1 - pred))

        # Focal modulation
        p_t = target * pred + (1 - target) * (1 - pred)
        focal_weight = (1 - p_t) ** self.gamma

        loss = focal_weight * bce
        return loss.mean()


# ── Training Loop ─────────────────────────────────────────────────────────────


def freeze_encoder(model: nn.Module) -> int:
    """Freeze encoder layers (down_blocks + bottleneck). Returns count of frozen params."""
    frozen = 0
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in ["down_block_", "bottleneck"]):
            param.requires_grad = False
            frozen += param.numel()
    return frozen


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True


def compute_recall(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    dist_threshold: float = 5.0,
) -> dict:
    """Compute recall and precision on a dataset.

    For each heatmap, threshold → find peak → compare to GT center.
    """
    model.eval()
    tp = 0
    fp = 0
    fn = 0
    total_error = 0.0
    n_correct = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            preds = model(inputs).cpu().numpy()
            targets_np = targets.numpy()

            for b in range(preds.shape[0]):
                for c in range(preds.shape[1]):
                    pred_hm = preds[b, c]
                    gt_hm = targets_np[b, c]

                    # GT: check if there's a ball
                    gt_has_ball = gt_hm.max() > 0.5
                    gt_y, gt_x = np.unravel_index(gt_hm.argmax(), gt_hm.shape) if gt_has_ball else (0, 0)

                    # Prediction: threshold and find peak
                    pred_has_ball = pred_hm.max() > threshold
                    if pred_has_ball:
                        pred_y, pred_x = np.unravel_index(pred_hm.argmax(), pred_hm.shape)
                    else:
                        pred_y, pred_x = 0, 0

                    if gt_has_ball and pred_has_ball:
                        dist = np.hypot(pred_x - gt_x, pred_y - gt_y)
                        if dist < dist_threshold:
                            tp += 1
                            total_error += dist
                            n_correct += 1
                        else:
                            fp += 1
                            fn += 1
                    elif gt_has_ball and not pred_has_ball:
                        fn += 1
                    elif not gt_has_ball and pred_has_ball:
                        fp += 1
                    # else: both empty → true negative

    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    mean_error = total_error / max(n_correct, 1)

    model.train()
    return {
        "recall": recall,
        "precision": precision,
        "f1": 2 * recall * precision / max(recall + precision, 1e-7),
        "tp": tp, "fp": fp, "fn": fn,
        "mean_error": mean_error,
    }


def train(args):
    """Main training function."""
    from app.pipeline.tracknet import TrackNet

    data_dir = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        log.info("GPU: %s, VRAM: %.1f GB",
                 torch.cuda.get_device_name(), torch.cuda.get_device_properties(0).total_memory / 1e9)

    # Load match directories from split files
    train_file = data_dir / "train.txt"
    val_file = data_dir / "val.txt"

    if not train_file.exists():
        log.error("train.txt not found in %s. Run prepare_tracknet_data.py first.", data_dir)
        sys.exit(1)

    train_matches = [data_dir / name.strip() for name in train_file.read_text().strip().split("\n")
                     if name.strip()]
    val_matches = [data_dir / name.strip() for name in val_file.read_text().strip().split("\n")
                   if name.strip()] if val_file.exists() else []

    log.info("Train matches: %d, Val matches: %d", len(train_matches), len(val_matches))

    # Create datasets
    train_dataset = TrackNetDataset(train_matches, seq_len=SEQ_LEN, augment=True)
    val_dataset = TrackNetDataset(val_matches, seq_len=SEQ_LEN, augment=False) if val_matches else None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    ) if val_dataset else None

    log.info("Train samples: %d, Val samples: %d",
             len(train_dataset), len(val_dataset) if val_dataset else 0)

    # Create model
    in_dim = (SEQ_LEN + 1) * 3  # 27 channels
    model = TrackNet(in_dim=in_dim, out_dim=SEQ_LEN)

    # Load pretrained weights
    start_epoch = 0
    if args.resume:
        log.info("Resuming from %s", args.resume)
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0) + 1
    elif args.pretrained:
        log.info("Loading pretrained weights: %s", args.pretrained)
        ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
    else:
        log.info("Training from scratch")

    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model parameters: %s", f"{n_params:,}")

    # Loss function
    loss_fn = WeightedFocalLoss(gamma=args.focal_gamma, pos_weight=args.pos_weight)

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6,
    )

    # Training loop
    best_recall = 0.0
    patience_counter = 0

    log.info("\n=== Training Start ===")
    log.info("Epochs: %d, Batch size: %d, LR: %.1e", args.epochs, args.batch_size, args.lr)
    log.info("Phase 1 (epochs 0-%d): Freeze encoder, LR=%.1e", args.phase1_epochs - 1, args.lr)
    log.info("Phase 2 (epochs %d+): Full fine-tune, LR=%.1e", args.phase1_epochs, args.lr / 10)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Phase control: freeze/unfreeze encoder
        if epoch < args.phase1_epochs:
            frozen = freeze_encoder(model)
            if epoch == 0:
                log.info("Phase 1: Encoder frozen (%s params)", f"{frozen:,}")
            # Recreate optimizer with only trainable params
            if epoch == start_epoch:
                optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=args.lr, weight_decay=args.weight_decay,
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=5, T_mult=2, eta_min=1e-6,
                )
        elif epoch == args.phase1_epochs:
            unfreeze_all(model)
            log.info("Phase 2: All layers unfrozen")
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr / 10, weight_decay=args.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=5, T_mult=2, eta_min=1e-6,
            )

        # Train one epoch
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(inputs)
            loss = loss_fn(preds, targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if batch_idx % 50 == 0 and batch_idx > 0:
                log.info("  Epoch %d [%d/%d] loss=%.4f lr=%.1e",
                         epoch, batch_idx, len(train_loader),
                         total_loss / n_batches, optimizer.param_groups[0]["lr"])

        scheduler.step()

        avg_loss = total_loss / max(n_batches, 1)
        epoch_time = time.time() - epoch_start

        # Validation
        val_metrics = {}
        if val_loader:
            val_metrics = compute_recall(model, val_loader, device, threshold=0.3, dist_threshold=10.0)

        # Log
        lr_current = optimizer.param_groups[0]["lr"]
        phase = "P1" if epoch < args.phase1_epochs else "P2"
        log.info(
            "Epoch %d/%d [%s] loss=%.4f lr=%.1e time=%.0fs"
            + (" | val recall=%.3f prec=%.3f f1=%.3f err=%.1fpx" if val_metrics else ""),
            epoch, args.epochs - 1, phase, avg_loss, lr_current, epoch_time,
            *([val_metrics["recall"], val_metrics["precision"],
               val_metrics["f1"], val_metrics["mean_error"]] if val_metrics else []),
        )

        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "param_dict": {
                "seq_len": SEQ_LEN,
                "bg_mode": "concat",
                "lr": lr_current,
                "batch_size": args.batch_size,
                "loss": avg_loss,
            },
        }
        if val_metrics:
            ckpt["val_metrics"] = val_metrics

        torch.save(ckpt, output_dir / "last.pt")

        # Save best by validation recall
        if val_metrics and val_metrics["recall"] > best_recall:
            best_recall = val_metrics["recall"]
            torch.save(ckpt, output_dir / "best.pt")
            log.info("  ★ New best recall: %.3f (saved best.pt)", best_recall)
            patience_counter = 0
        elif val_metrics:
            patience_counter += 1

        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(ckpt, output_dir / f"epoch_{epoch}.pt")

        # Early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            log.info("Early stopping: no improvement for %d epochs", args.patience)
            break

    log.info("\n=== Training Complete ===")
    log.info("Best validation recall: %.3f", best_recall)
    log.info("Checkpoints saved to: %s", output_dir)
    log.info("  best.pt  — best validation recall")
    log.info("  last.pt  — last epoch")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune TrackNet")
    parser.add_argument("--data", type=str, default="data/tracknet_training",
                        help="Training data directory (output of prepare_tracknet_data.py)")
    parser.add_argument("--pretrained", type=str, default="model_weight/TrackNet_best.pt",
                        help="Pretrained TrackNet weights")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--output", type=str, default="model_weight/tracknet_finetuned",
                        help="Output directory for checkpoints")

    # Training params
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=4,
                        help="Early stopping patience (0=disabled)")

    # Phase control
    parser.add_argument("--phase1-epochs", type=int, default=5,
                        help="Number of epochs to freeze encoder")

    # Loss params
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--pos-weight", type=float, default=50.0,
                        help="Positive class weight for focal loss")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
