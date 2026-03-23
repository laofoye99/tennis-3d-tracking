"""Canonical-ray event classifier for tennis ball tracking.

Classifies each frame as: bounce / hit / serve / fly
Input: sliding window of dual-camera 2D pixel observations (canonical ray style).
       No 3D reconstruction needed — network learns 2D → event mapping directly.
Model: BiLSTM on normalized pixel sequences.

Training data: synthetic trajectories from Unity (with cam66/cam68 pixel projections).
Inference: real 2D detections from TrackNet (no domain gap on pixel level).
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# Event labels
LABELS = {"fly": 0, "bounce": 1, "hit": 2, "serve": 3}
LABEL_NAMES = {v: k for k, v in LABELS.items()}
NUM_CLASSES = len(LABELS)

# Window size (frames before + after center frame)
WINDOW_HALF = 12  # 12 frames each side = 25 total at 25fps = 1 second
WINDOW_SIZE = 2 * WINDOW_HALF + 1


class TrajectoryEventDataset(Dataset):
    """Extract sliding windows from noisy trajectory files."""

    def __init__(self, json_paths: list, oversample_events: bool = True):
        """Load trajectories and extract windows.

        Args:
            json_paths: List of noisy trajectory JSON files.
            oversample_events: If True, oversample bounce/hit/serve to balance classes.
        """
        self.windows = []  # (features, label)
        self._load_all(json_paths)

        if oversample_events:
            self._oversample()

        logger.info("Dataset: %d windows (fly=%d, bounce=%d, hit=%d, serve=%d)",
                     len(self.windows),
                     sum(1 for _, l in self.windows if l == 0),
                     sum(1 for _, l in self.windows if l == 1),
                     sum(1 for _, l in self.windows if l == 2),
                     sum(1 for _, l in self.windows if l == 3))

    def _load_all(self, paths):
        for p in paths:
            with open(p) as f:
                data = json.load(f)
            frames = data["frames"]
            self._extract_windows(frames)

    def _extract_windows(self, frames):
        n = len(frames)
        if n < WINDOW_SIZE:
            return

        evts = [LABELS.get(f.get("evt", "fly"), 0) for f in frames]

        # Dual-camera pixel coordinates (normalized to [0,1])
        IMG_W, IMG_H = 1920.0, 1080.0
        c66 = np.array([f.get("c66", [0, 0, 0])[:2] for f in frames], dtype=np.float32)
        c68 = np.array([f.get("c68", [0, 0, 0])[:2] for f in frames], dtype=np.float32)
        c66[:, 0] /= IMG_W; c66[:, 1] /= IMG_H
        c68[:, 0] /= IMG_W; c68[:, 1] /= IMG_H

        # Pixel velocity (frame-to-frame displacement, captures motion direction)
        dc66 = np.zeros_like(c66)
        dc66[1:] = (c66[1:] - c66[:-1]) * 25.0  # normalized px/s
        dc68 = np.zeros_like(c68)
        dc68[1:] = (c68[1:] - c68[:-1]) * 25.0

        # Pixel acceleration (direction change = bounce/hit signature)
        ac66 = np.zeros_like(c66)
        ac66[1:] = (dc66[1:] - dc66[:-1]) * 25.0
        ac68 = np.zeros_like(c68)
        ac68[1:] = (dc68[1:] - dc68[:-1]) * 25.0

        # Pixel speed magnitude per camera
        spd66 = np.linalg.norm(dc66, axis=1, keepdims=True)
        spd68 = np.linalg.norm(dc68, axis=1, keepdims=True)

        # Features: c66(2) + c68(2) + dc66(2) + dc68(2) + ac66(2) + ac68(2) + spd66(1) + spd68(1) = 14
        features = np.concatenate([c66, c68, dc66, dc68, ac66, ac68, spd66, spd68], axis=1)

        for i in range(WINDOW_HALF, n - WINDOW_HALF):
            window = features[i - WINDOW_HALF:i + WINDOW_HALF + 1]  # (25, 14)
            label = evts[i]
            self.windows.append((window, label))

    def _oversample(self):
        """Downsample fly + oversample events for balanced training."""
        by_class = {c: [] for c in range(NUM_CLASSES)}
        for w, l in self.windows:
            by_class[l].append((w, l))

        # Target count: 5x the largest event class
        event_max = max(len(by_class[c]) for c in range(1, NUM_CLASSES))
        target = event_max * 5

        # Downsample fly to target
        fly_samples = by_class[0]
        if len(fly_samples) > target:
            indices = np.random.choice(len(fly_samples), target, replace=False)
            fly_samples = [fly_samples[i] for i in indices]

        balanced = list(fly_samples)
        for c in range(1, NUM_CLASSES):
            samples = by_class[c]
            if not samples:
                continue
            # Oversample to match target
            n_repeat = max(1, target // len(samples))
            remainder = target - len(samples) * n_repeat
            balanced.extend(samples * n_repeat)
            if remainder > 0:
                indices = np.random.choice(len(samples), remainder, replace=True)
                balanced.extend([samples[i] for i in indices])

        np.random.shuffle(balanced)
        self.windows = balanced

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window, label = self.windows[idx]
        return torch.from_numpy(window), label


class EventClassifierNet(nn.Module):
    """BiLSTM for trajectory event classification.

    Input: (batch, window_size=25, features=13)
    Output: (batch, num_classes=4)

    ~5K parameters. Minimal BiLSTM, suited for small datasets.
    """

    def __init__(self, in_features=14, hidden_size=24, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, NUM_CLASSES),
        )

    def forward(self, x):
        # x: (B, T, F) where T=25, F=13
        h, _ = self.lstm(x)  # (B, T, hidden*2)

        # Take center frame's representation
        center = h[:, WINDOW_HALF, :]  # (B, hidden*2)

        return self.head(center)  # (B, NUM_CLASSES)


def train_model(
    train_paths: list,
    val_paths: list = None,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cuda",
    save_path: str = "model_weight/event_classifier.pt",
):
    """Train the event classifier."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info("Training on %s", device)

    # Data
    train_ds = TrajectoryEventDataset(train_paths, oversample_events=True)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=True)

    val_dl = None
    if val_paths:
        val_ds = TrajectoryEventDataset(val_paths, oversample_events=False)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = EventClassifierNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %d parameters", total_params)

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Class weights — moderate balance, not too aggressive
    weights = torch.tensor([0.5, 2.0, 2.0, 2.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_val_f1 = 0
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_dl:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            train_correct += (logits.argmax(1) == batch_y).sum().item()
            train_total += batch_x.size(0)

        scheduler.step()
        train_acc = train_correct / train_total

        # Validate
        val_str = ""
        if val_dl is not None:
            val_metrics = evaluate(model, val_dl, device)
            val_str = f" | val_acc={val_metrics['accuracy']:.3f} val_f1={val_metrics['macro_f1']:.3f}"

            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "val_f1": best_val_f1}, save_path)

        logger.info("Epoch %2d/%d: loss=%.4f acc=%.3f%s",
                     epoch + 1, epochs,
                     train_loss / train_total, train_acc, val_str)

    # Save final if no val
    if val_dl is None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "epoch": epochs}, save_path)

    logger.info("Saved model to %s", save_path)
    return model


def evaluate(model, dataloader, device):
    """Evaluate model, return per-class metrics."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()

    # Per-class precision, recall, F1
    metrics = {"accuracy": float(accuracy)}
    f1s = []
    for c in range(NUM_CLASSES):
        tp = ((all_preds == c) & (all_labels == c)).sum()
        fp = ((all_preds == c) & (all_labels != c)).sum()
        fn = ((all_preds != c) & (all_labels == c)).sum()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        name = LABEL_NAMES[c]
        metrics[f"{name}_precision"] = float(prec)
        metrics[f"{name}_recall"] = float(rec)
        metrics[f"{name}_f1"] = float(f1)
        if c > 0:  # exclude fly from macro F1
            f1s.append(f1)

    metrics["macro_f1"] = float(np.mean(f1s)) if f1s else 0
    return metrics


def predict(model, trajectory_3d: dict, device="cuda"):
    """Run inference on a real 3D trajectory.

    Args:
        model: Trained EventClassifierNet.
        trajectory_3d: dict of frame_idx -> (x, y, z).
        device: torch device.

    Returns:
        dict of frame_idx -> {"label": str, "probs": dict}.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    frames = sorted(trajectory_3d.keys())
    if len(frames) < WINDOW_SIZE:
        return {}

    # Build position array
    pos = np.array([trajectory_3d[f][:3] for f in frames], dtype=np.float32)

    # Compute vel, acc
    vel = np.zeros_like(pos)
    vel[1:] = (pos[1:] - pos[:-1]) * 25.0
    acc = np.zeros_like(pos)
    acc[1:] = (vel[1:] - vel[:-1]) * 25.0
    speed = np.linalg.norm(vel, axis=1, keepdims=True)
    z = pos[:, 2:3]
    vz = vel[:, 2:3]
    az = acc[:, 2:3]
    features = np.concatenate([pos, vel, acc, speed, z, vz, az], axis=1)

    # Predict each frame
    results = {}
    with torch.no_grad():
        for i in range(WINDOW_HALF, len(frames) - WINDOW_HALF):
            window = features[i - WINDOW_HALF:i + WINDOW_HALF + 1]
            x = torch.from_numpy(window).unsqueeze(0).to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            pred = int(probs.argmax())
            results[frames[i]] = {
                "label": LABEL_NAMES[pred],
                "probs": {LABEL_NAMES[c]: float(probs[c]) for c in range(NUM_CLASSES)},
            }

    return results
