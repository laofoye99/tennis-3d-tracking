"""Train rally segmentation model.

Classifies each frame as in_rally (match_ball) or not_rally using
features derived from cached TrackNet detections and GT labels.

Usage:
    python -m tools.train_rally_model
"""

import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GT_DIR = "uploads/cam66_20260307_173403_2min"
CACHE_PATH = Path(".cache/detection_cache.pkl")
MODEL_DIR = Path("model_weight")
MAX_FRAMES = 1800


# ── GT loading ───────────────────────────────────────────────────────


def load_gt_labels(gt_dir: str, max_frames: int) -> np.ndarray:
    """Return binary array: 1 = match_ball (rally), 0 = not rally."""
    labels = np.zeros(max_frames, dtype=np.int32)
    for fi in range(max_frames):
        fp = os.path.join(gt_dir, f"{fi:05d}.json")
        if not os.path.exists(fp):
            continue
        with open(fp) as f:
            data = json.load(f)
        for s in data.get("shapes", []):
            if s.get("label", "") != "ball":
                continue
            desc = s.get("description", "").lower().replace("\uff0c", ",")
            if "match_ball" in desc:
                labels[fi] = 1
                break
    return labels


# ── Feature extraction ───────────────────────────────────────────────


def extract_features(det66: dict, multi66: dict, multi68: dict,
                     max_frames: int) -> np.ndarray:
    """Build per-frame feature vectors from detection data.

    Multi-scale windows with trajectory-aware features.
    """
    # Pre-build arrays
    has_det66 = np.zeros(max_frames, dtype=bool)
    has_det68 = np.zeros(max_frames, dtype=bool)
    px66 = np.full(max_frames, np.nan)
    py66 = np.full(max_frames, np.nan)
    px68 = np.full(max_frames, np.nan)
    py68 = np.full(max_frames, np.nan)
    blob_sum66 = np.zeros(max_frames)
    blob_max66 = np.zeros(max_frames)
    blob_area66 = np.zeros(max_frames)
    n_blobs66 = np.zeros(max_frames)

    for fi, (x, y, bsum) in det66.items():
        if fi < max_frames:
            has_det66[fi] = True
            px66[fi] = x
            py66[fi] = y
            blob_sum66[fi] = bsum

    for fi, blobs in multi66.items():
        if fi < max_frames and blobs:
            blob_max66[fi] = blobs[0].get("blob_max", 0)
            blob_area66[fi] = blobs[0].get("blob_area", 0)
            n_blobs66[fi] = len(blobs)

    for fi, blobs in multi68.items():
        if fi < max_frames and blobs:
            has_det68[fi] = True
            px68[fi] = blobs[0]["pixel_x"]
            py68[fi] = blobs[0]["pixel_y"]

    # Per-frame velocity
    vel66 = np.zeros(max_frames)
    vel_x = np.zeros(max_frames)
    vel_y = np.zeros(max_frames)
    for fi in range(1, max_frames):
        if has_det66[fi] and has_det66[fi - 1]:
            dx = px66[fi] - px66[fi - 1]
            dy = py66[fi] - py66[fi - 1]
            vel_x[fi] = dx
            vel_y[fi] = dy
            vel66[fi] = np.sqrt(dx * dx + dy * dy)

    # Acceleration
    accel66 = np.abs(np.diff(vel66, prepend=0))

    # Gap features
    gap_before = np.zeros(max_frames)
    gap_after = np.zeros(max_frames)
    last_det = -999
    for fi in range(max_frames):
        if has_det66[fi]:
            last_det = fi
        gap_before[fi] = fi - last_det
    last_det = max_frames + 999
    for fi in range(max_frames - 1, -1, -1):
        if has_det66[fi]:
            last_det = fi
        gap_after[fi] = last_det - fi

    # Y-direction changes (key rally feature: ball goes up and down)
    y_sign_changes = np.zeros(max_frames)
    for fi in range(2, max_frames):
        if has_det66[fi] and has_det66[fi - 1] and has_det66[fi - 2]:
            dy_prev = py66[fi - 1] - py66[fi - 2]
            dy_curr = py66[fi] - py66[fi - 1]
            if dy_prev * dy_curr < 0:
                y_sign_changes[fi] = 1

    # Consecutive detection run length (how many consecutive frames have detection)
    run_length = np.zeros(max_frames)
    curr_run = 0
    for fi in range(max_frames):
        if has_det66[fi]:
            curr_run += 1
        else:
            curr_run = 0
        run_length[fi] = curr_run

    # Backward run length
    run_length_back = np.zeros(max_frames)
    curr_run = 0
    for fi in range(max_frames - 1, -1, -1):
        if has_det66[fi]:
            curr_run += 1
        else:
            curr_run = 0
        run_length_back[fi] = curr_run

    # Total run length (forward + backward - 1)
    total_run = np.zeros(max_frames)
    for fi in range(max_frames):
        if has_det66[fi]:
            total_run[fi] = run_length[fi] + run_length_back[fi] - 1

    all_features = []

    # Multi-scale window features
    for w in [10, 25, 50, 100]:
        half = w // 2
        feat_density = np.zeros(max_frames)
        feat_mean_bsum = np.zeros(max_frames)
        feat_mean_bmax = np.zeros(max_frames)
        feat_mean_vel = np.zeros(max_frames)
        feat_std_vel = np.zeros(max_frames)
        feat_xcam = np.zeros(max_frames)
        feat_y_range = np.zeros(max_frames)
        feat_x_range = np.zeros(max_frames)
        feat_regularity = np.zeros(max_frames)
        feat_y_sign_count = np.zeros(max_frames)
        feat_mean_area = np.zeros(max_frames)
        feat_pos_std_x = np.zeros(max_frames)
        feat_pos_std_y = np.zeros(max_frames)
        feat_mean_gap = np.zeros(max_frames)
        feat_max_gap = np.zeros(max_frames)
        feat_det68_density = np.zeros(max_frames)

        for fi in range(max_frames):
            lo = max(0, fi - half)
            hi = min(max_frames, fi + half + 1)
            win_len = hi - lo

            det_count = has_det66[lo:hi].sum()
            feat_density[fi] = det_count / win_len
            feat_det68_density[fi] = has_det68[lo:hi].sum() / win_len

            if det_count > 0:
                mask = has_det66[lo:hi]
                feat_mean_bsum[fi] = blob_sum66[lo:hi][mask].mean()
                feat_mean_bmax[fi] = blob_max66[lo:hi][mask].mean()
                feat_mean_area[fi] = blob_area66[lo:hi][mask].mean()

            feat_mean_vel[fi] = vel66[lo:hi].mean()
            if det_count > 1:
                feat_std_vel[fi] = vel66[lo:hi][has_det66[lo:hi]].std()

            both = (has_det66[lo:hi] & has_det68[lo:hi]).sum()
            feat_xcam[fi] = both / max(det_count, 1)

            valid_x = px66[lo:hi][~np.isnan(px66[lo:hi])]
            valid_y = py66[lo:hi][~np.isnan(py66[lo:hi])]
            if len(valid_x) > 1:
                feat_y_range[fi] = valid_y.max() - valid_y.min()
                feat_x_range[fi] = valid_x.max() - valid_x.min()
                feat_pos_std_x[fi] = np.std(valid_x)
                feat_pos_std_y[fi] = np.std(valid_y)

            # Y-direction change count in window
            feat_y_sign_count[fi] = y_sign_changes[lo:hi].sum()

            # Inter-detection gap statistics
            det_indices = np.where(has_det66[lo:hi])[0]
            if len(det_indices) > 1:
                intervals = np.diff(det_indices)
                feat_regularity[fi] = np.std(intervals)
                feat_mean_gap[fi] = np.mean(intervals)
                feat_max_gap[fi] = np.max(intervals)
            elif det_count <= 1:
                feat_regularity[fi] = w
                feat_mean_gap[fi] = w
                feat_max_gap[fi] = w

        all_features.extend([
            feat_density, feat_mean_bsum, feat_mean_bmax, feat_mean_vel,
            feat_std_vel, feat_xcam, feat_y_range, feat_x_range,
            feat_regularity, feat_y_sign_count, feat_mean_area,
            feat_pos_std_x, feat_pos_std_y, feat_mean_gap, feat_max_gap,
            feat_det68_density,
        ])

    # Frame-level features
    all_features.extend([
        gap_before, gap_after,
        np.minimum(gap_before, gap_after),  # min gap (proximity to nearest det)
        blob_sum66, blob_max66, blob_area66,
        has_det66.astype(float), has_det68.astype(float),
        (has_det66 & has_det68).astype(float),  # both cameras detect
        vel66, vel_x, vel_y, accel66,
        n_blobs66, run_length, total_run,
    ])

    # Rolling detection rate at multiple scales
    cum_det = np.cumsum(has_det66.astype(float))
    for lookback in [25, 50, 100, 200]:
        rate = np.zeros(max_frames)
        for fi in range(max_frames):
            lo = max(0, fi - lookback)
            rate[fi] = (cum_det[fi] - (cum_det[lo - 1] if lo > 0 else 0)) / (fi - lo + 1)
        all_features.append(rate)

    # Position features (absolute, normalized)
    px_norm = px66.copy()
    py_norm = py66.copy()
    px_norm[np.isnan(px_norm)] = 0
    py_norm[np.isnan(py_norm)] = 0
    all_features.extend([px_norm / 1920.0, py_norm / 1080.0])

    # Fourier features of detection pattern (captures periodicity)
    det_signal = has_det66.astype(float)
    for freq_win in [50, 100]:
        power_at_freq = np.zeros(max_frames)
        for fi in range(max_frames):
            lo = max(0, fi - freq_win // 2)
            hi = min(max_frames, fi + freq_win // 2)
            seg = det_signal[lo:hi]
            if len(seg) > 4:
                fft = np.fft.rfft(seg - seg.mean())
                power = np.abs(fft) ** 2
                # Sum power in "rally frequency" range (2-10 Hz at 25fps = indices 4-20 for win=50)
                lo_idx = max(1, len(power) // 10)
                hi_idx = min(len(power), len(power) // 2)
                if hi_idx > lo_idx:
                    power_at_freq[fi] = power[lo_idx:hi_idx].sum() / power.sum() if power.sum() > 0 else 0
        all_features.append(power_at_freq)

    X = np.column_stack(all_features)
    return X


# ── Smoothing post-process ──────────────────────────────────────────


def smooth_predictions(preds: np.ndarray, min_rally: int = 5,
                       min_gap: int = 10) -> np.ndarray:
    """Temporal smoothing: remove short rally segments and fill short gaps."""
    result = preds.copy()
    n = len(result)

    # Fill short gaps in rally (< min_gap frames of 0 between rally segments)
    i = 0
    while i < n:
        if result[i] == 0:
            j = i
            while j < n and result[j] == 0:
                j += 1
            gap_len = j - i
            # If this gap is short AND surrounded by rally on both sides
            if gap_len < min_gap and i > 0 and j < n:
                if result[i - 1] == 1 and result[j] == 1:
                    result[i:j] = 1
            i = j
        else:
            i += 1

    # Remove short rally segments (< min_rally frames)
    i = 0
    while i < n:
        if result[i] == 1:
            j = i
            while j < n and result[j] == 1:
                j += 1
            if j - i < min_rally:
                result[i:j] = 0
            i = j
        else:
            i += 1

    return result


# ── Training ─────────────────────────────────────────────────────────


def train_random_forest(X_train, y_train, X_val, y_val):
    logger.info("Training Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=1000,
        max_depth=15,
        min_samples_leaf=2,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_val)[:, 1]
    return clf, y_prob


def train_gradient_boosting(X_train, y_train, X_val, y_val):
    logger.info("Training Gradient Boosting...")
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale = n_neg / max(n_pos, 1)

    try:
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=800,
            max_depth=7,
            learning_rate=0.03,
            scale_pos_weight=scale,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
        )
        logger.info("Using XGBoost")
    except ImportError:
        clf = GradientBoostingClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            random_state=42,
        )
        logger.info("Using sklearn GradientBoosting")

    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_val)[:, 1]
    return clf, y_prob


def train_bilstm(X_train, y_train, X_val, y_val):
    """Train BiLSTM classifier with PyTorch."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        logger.warning("PyTorch not available, skipping BiLSTM")
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training BiLSTM on %s...", device)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    seq_len = 51
    n_features = X_train_s.shape[1]

    def make_sequences(X, y):
        pad = seq_len // 2
        X_pad = np.pad(X, ((pad, pad), (0, 0)), mode="edge")
        seqs = np.array([X_pad[i:i + seq_len] for i in range(len(X))])
        return seqs, y

    X_seq_train, y_seq_train = make_sequences(X_train_s, y_train)
    X_seq_val, y_seq_val = make_sequences(X_val_s, y_val)

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)

    class BiLSTMClassifier(nn.Module):
        def __init__(self, input_size, hidden_size=96, num_layers=2, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers=num_layers,
                batch_first=True, bidirectional=True, dropout=dropout,
            )
            self.attn = nn.Linear(hidden_size * 2, 1)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size * 2, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            # Attention over all timesteps
            attn_w = torch.softmax(self.attn(out), dim=1)
            context = (attn_w * out).sum(dim=1)
            return self.fc(context).squeeze(-1)

    model = BiLSTMClassifier(n_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)

    train_ds = TensorDataset(
        torch.FloatTensor(X_seq_train), torch.FloatTensor(y_seq_train),
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_seq_val), torch.FloatTensor(y_seq_val),
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    best_f1 = 0
    best_state = None
    patience = 20
    no_improve = 0

    for epoch in range(120):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                probs = torch.sigmoid(model(xb)).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(yb.numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        preds = (all_probs > 0.5).astype(int)
        val_f1 = f1_score(all_labels, preds)
        val_acc = accuracy_score(all_labels, preds)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_probs = all_probs.copy()
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0:
            logger.info(
                "Epoch %d: loss=%.4f val_acc=%.3f val_f1=%.3f (best=%.3f)",
                epoch, total_loss / len(train_loader), val_acc, val_f1, best_f1,
            )

        if no_improve >= patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    # Save model + scaler
    save_data = {
        "model_state": best_state,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "n_features": n_features,
        "seq_len": seq_len,
    }
    MODEL_DIR.mkdir(exist_ok=True)

    import torch as _torch
    _torch.save(save_data, MODEL_DIR / "rally_segmentation.pt")
    logger.info("Saved BiLSTM model to %s", MODEL_DIR / "rally_segmentation.pt")

    return model, best_probs


# ── Evaluation ───────────────────────────────────────────────────────


def find_best_threshold(y_true, y_prob):
    """Find threshold that maximizes accuracy."""
    best_acc = 0
    best_thr = 0.5
    for thr in np.arange(0.1, 0.9, 0.02):
        preds = (y_prob >= thr).astype(int)
        acc = accuracy_score(y_true, preds)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    return best_thr, best_acc


def evaluate(name: str, y_true: np.ndarray, y_prob: np.ndarray):
    """Evaluate with threshold tuning and smoothing."""
    # Default threshold
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Find best threshold
    best_thr, best_acc = find_best_threshold(y_true, y_prob)
    y_pred_thr = (y_prob >= best_thr).astype(int)
    f1_thr = f1_score(y_true, y_pred_thr)

    # Apply smoothing to threshold-tuned predictions
    y_pred_smooth = smooth_predictions(y_pred_thr, min_rally=5, min_gap=8)
    acc_smooth = accuracy_score(y_true, y_pred_smooth)
    f1_smooth = f1_score(y_true, y_pred_smooth)

    cm = confusion_matrix(y_true, y_pred_smooth)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Default (thr=0.50): acc={acc:.4f} f1={f1:.4f}")
    print(f"  Best threshold={best_thr:.2f}: acc={best_acc:.4f} f1={f1_thr:.4f}")
    print(f"  + Smoothing: acc={acc_smooth:.4f} f1={f1_smooth:.4f}")
    print(f"\n  Confusion Matrix (smoothed):")
    print(f"    TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"    FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    print(classification_report(
        y_true, y_pred_smooth, target_names=["not_rally", "rally"],
    ))

    return acc_smooth, f1_smooth, best_thr


# ── Main ─────────────────────────────────────────────────────────────


def main():
    logger.info("Loading GT labels...")
    labels = load_gt_labels(GT_DIR, MAX_FRAMES)
    n_rally = labels.sum()
    logger.info("GT: %d rally / %d total (%.1f%%)", n_rally, MAX_FRAMES,
                100 * n_rally / MAX_FRAMES)

    logger.info("Loading detection cache...")
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)

    det66 = cache["det66"]
    multi66 = cache["multi66"]
    multi68 = cache["multi68"]

    logger.info("Extracting features...")
    X = extract_features(det66, multi66, multi68, MAX_FRAMES)
    y = labels
    logger.info("Feature matrix: %s", X.shape)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Block-based stratified split to reduce temporal leakage
    block_size = 25
    n_blocks = MAX_FRAMES // block_size
    block_labels = np.array([
        labels[i * block_size:(i + 1) * block_size].max() for i in range(n_blocks)
    ])
    block_indices = np.arange(n_blocks)

    train_blocks, val_blocks = train_test_split(
        block_indices, test_size=0.3, random_state=42, stratify=block_labels,
    )

    train_frames = np.concatenate([
        np.arange(b * block_size, (b + 1) * block_size) for b in sorted(train_blocks)
    ])
    val_frames = np.concatenate([
        np.arange(b * block_size, (b + 1) * block_size) for b in sorted(val_blocks)
    ])

    X_train, y_train = X[train_frames], y[train_frames]
    X_val, y_val = X[val_frames], y[val_frames]

    logger.info("Train: %d frames (%d rally), Val: %d frames (%d rally)",
                len(y_train), y_train.sum(), len(y_val), y_val.sum())

    # ── Model 1: Random Forest ──
    rf_model, rf_prob = train_random_forest(X_train, y_train, X_val, y_val)
    rf_acc, rf_f1, rf_thr = evaluate("Random Forest", y_val, rf_prob)

    # ── Model 2: Gradient Boosting ──
    gb_model, gb_prob = train_gradient_boosting(X_train, y_train, X_val, y_val)
    gb_acc, gb_f1, gb_thr = evaluate("Gradient Boosting", y_val, gb_prob)

    # ── Model 3: BiLSTM ──
    lstm_model, lstm_prob = train_bilstm(X_train, y_train, X_val, y_val)
    if lstm_prob is not None:
        lstm_acc, lstm_f1, lstm_thr = evaluate("BiLSTM", y_val, lstm_prob)
    else:
        lstm_acc, lstm_f1 = 0, 0

    # ── Ensemble: average probabilities ──
    if lstm_prob is not None:
        ens_prob = (rf_prob + gb_prob + lstm_prob) / 3
    else:
        ens_prob = (rf_prob + gb_prob) / 2
    ens_acc, ens_f1, ens_thr = evaluate("Ensemble", y_val, ens_prob)

    # ── Pick best ──
    results = {
        "Random Forest": (rf_acc, rf_f1, rf_model, rf_thr, rf_prob),
        "Gradient Boosting": (gb_acc, gb_f1, gb_model, gb_thr, gb_prob),
        "Ensemble": (ens_acc, ens_f1, None, ens_thr, ens_prob),
    }
    if lstm_prob is not None:
        results["BiLSTM"] = (lstm_acc, lstm_f1, lstm_model, lstm_thr, lstm_prob)

    best_name = max(results, key=lambda k: results[k][0])
    best_acc = results[best_name][0]

    print(f"\n{'#'*60}")
    print(f"  Best model: {best_name} (accuracy: {best_acc:.4f})")
    print(f"{'#'*60}")

    # Save best sklearn model if applicable
    MODEL_DIR.mkdir(exist_ok=True)
    if best_name in ("Random Forest", "Gradient Boosting"):
        save_path = MODEL_DIR / "rally_segmentation.pkl"
        with open(save_path, "wb") as f:
            pickle.dump({
                "model": results[best_name][2],
                "model_name": best_name,
                "threshold": results[best_name][3],
            }, f)
        logger.info("Saved best model to %s", save_path)
    elif best_name == "Ensemble":
        # Save both sklearn models for ensemble
        save_path = MODEL_DIR / "rally_segmentation.pkl"
        with open(save_path, "wb") as f:
            pickle.dump({
                "rf_model": rf_model,
                "gb_model": gb_model,
                "model_name": "Ensemble",
                "threshold": ens_thr,
            }, f)
        logger.info("Saved ensemble models to %s", save_path)

    # Feature importance
    if hasattr(gb_model, "feature_importances_"):
        imp = gb_model.feature_importances_
        top_k = min(15, len(imp))
        top_idx = np.argsort(imp)[-top_k:][::-1]
        print(f"\nTop {top_k} GBT feature importances:")
        for i, idx in enumerate(top_idx):
            print(f"  {i+1}. Feature {idx}: {imp[idx]:.4f}")

    if best_acc >= 0.90:
        logger.info("TARGET MET: %.1f%% accuracy (>= 90%%)", best_acc * 100)
    else:
        logger.warning("Target NOT met: %.1f%% < 90%%. Iterating...", best_acc * 100)

    return best_acc


if __name__ == "__main__":
    main()
