#!/usr/bin/env python3


import os
import argparse
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); from scripts.evaluation import calculate_eer


# ------------------------- Repro ------------------------- #
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------- SpecAugment ------------------------- #
def time_mask(x: torch.Tensor, max_width: int, num_masks: int) -> torch.Tensor:
    """
    x: (C, T) or (B, C, T)  (we use (C,T) per sample here)
    """
    if max_width <= 0 or num_masks <= 0:
        return x
    C, T = x.shape
    for _ in range(num_masks):
        w = random.randint(0, min(max_width, T))
        if w == 0:
            continue
        t0 = random.randint(0, max(0, T - w))
        x[:, t0:t0 + w] = 0.0
    return x


def freq_mask(x: torch.Tensor, max_width: int, num_masks: int) -> torch.Tensor:
    """
    x: (C, T)
    """
    if max_width <= 0 or num_masks <= 0:
        return x
    C, T = x.shape
    for _ in range(num_masks):
        w = random.randint(0, min(max_width, C))
        if w == 0:
            continue
        c0 = random.randint(0, max(0, C - w))
        x[c0:c0 + w, :] = 0.0
    return x


# ------------------------- Dataset ------------------------- #
class AudioDataset(Dataset):
    def __init__(self, features_pkl: str, labels_pkl: Optional[str] = None):
        self.features_df = pd.read_pickle(features_pkl)
        self.has_labels = labels_pkl is not None and os.path.exists(labels_pkl)
        self.labels_df = pd.read_pickle(labels_pkl) if self.has_labels else None

        # Ensure we can map uttid -> label
        if self.has_labels:
            self.label_map = dict(zip(self.labels_df["uttid"].tolist(), self.labels_df["label"].tolist()))
        else:
            self.label_map = None

        # Basic sanity
        assert "uttid" in self.features_df.columns and "features" in self.features_df.columns

    def __len__(self):
        return len(self.features_df)

    def __getitem__(self, idx: int):
        row = self.features_df.iloc[idx]
        uttid = row["uttid"]
        feat = row["features"]  # torch.Tensor [C, T]
        if not torch.is_tensor(feat):
            feat = torch.tensor(feat)

        if self.has_labels:
            y = int(self.label_map[uttid])
            return uttid, feat, y
        else:
            return uttid, feat, None


def collate_fn(batch):
    uttids = [b[0] for b in batch]
    feats = [b[1].transpose(0, 1) for b in batch]  # (T, C) for pad_sequence
    feats_pad = pad_sequence(feats, batch_first=True)  # (B, T, C)
    feats_pad = feats_pad.transpose(1, 2).contiguous()  # (B, C, T)
    lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)

    labels = [b[2] for b in batch]
    if labels[0] is None:
        y = None
    else:
        y = torch.tensor(labels, dtype=torch.float32)  # BCEWithLogits expects float targets

    return uttids, feats_pad, lengths, y


# ------------------------- Model ------------------------- #
class StatsPool(nn.Module):
    """Mean+Std pooling over time (masked) -> robust utterance embedding."""
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        B, C, T = x.shape
        device = x.device
        mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)  # (B, T)
        mask = mask.unsqueeze(1).float()  # (B, 1, T)

        denom = mask.sum(dim=2).clamp(min=1.0)  # (B,1)
        mean = (x * mask).sum(dim=2) / denom  # (B,C)

        var = (mask * (x - mean.unsqueeze(-1)) ** 2).sum(dim=2) / denom
        std = torch.sqrt(var.clamp(min=1e-6))
        return torch.cat([mean, std], dim=1)  # (B, 2C)


class ConvEncoder(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepfakeDetector(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.enc = ConvEncoder(in_ch=in_ch, hidden=hidden, dropout=dropout)
        self.pool = StatsPool()
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        h = self.enc(x)
        z = self.pool(h, lengths)
        logits = self.head(z).squeeze(1)  # (B,)
        return logits


# ------------------------- EMA ------------------------- #
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            assert n in self.shadow
            self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        self.backup = {}
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.backup[n] = p.detach().clone()
            p.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model: nn.Module):
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            p.copy_(self.backup[n])
        self.backup = {}


# ------------------------- Train/Eval ------------------------- #
@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_prob: bool = False
) -> pd.DataFrame:
    model.eval()
    all_ids: List[str] = []
    all_scores: List[float] = []
    for uttids, x, lengths, _ in tqdm(loader, desc="Predict", leave=False):
        x = x.to(device)
        lengths = lengths.to(device)
        logits = model(x, lengths)
        if use_prob:
            scores = torch.sigmoid(logits)
        else:
            scores = logits
        all_ids.extend(uttids)
        all_scores.extend(scores.detach().cpu().numpy().tolist())

    return pd.DataFrame({"uttid": all_ids, "predictions": all_scores})


@torch.no_grad()
def evaluate_eer(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> float:
    model.eval()
    y_true = []
    y_score = []
    for _, x, lengths, y in tqdm(loader, desc="Eval", leave=False):
        x = x.to(device)
        lengths = lengths.to(device)
        logits = model(x, lengths)
        y_true.extend(y.numpy().tolist())
        y_score.extend(logits.detach().cpu().numpy().tolist())
    eer, _thr = calculate_eer(np.array(y_score), np.array(y_true))
    return float(eer)


def compute_class_weights(labels: np.ndarray) -> Tuple[float, float]:
    # labels are 0/1
    pos = (labels == 1).sum()
    neg = (labels == 0).sum()
    # pos_weight in BCEWithLogitsLoss is (neg/pos)
    pos_weight = (neg / max(pos, 1))
    # Also return class sampling weights
    w0 = 1.0 / max(neg, 1)
    w1 = 1.0 / max(pos, 1)
    return float(pos_weight), float(w0), float(w1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--dev_split", default="dev")
    ap.add_argument("--test_split", default="test2")  # mts / test2 / test1 / dev
    ap.add_argument("--ckpt_path", default="best_model.pth")
    ap.add_argument("--prediction_pkl", default="prediction.pkl")

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=5.0)

    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)

    ap.add_argument("--use_prob", action="store_true", help="save sigmoid(prob) instead of logits")

    # SpecAugment
    ap.add_argument("--specaug", action="store_true", help="enable SpecAugment during training")
    ap.add_argument("--time_mask_max", type=int, default=30)
    ap.add_argument("--time_mask_n", type=int, default=2)
    ap.add_argument("--freq_mask_max", type=int, default=24)
    ap.add_argument("--freq_mask_n", type=int, default=2)

    # EMA + early stop
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--patience", type=int, default=6)

    args = ap.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)

    # Paths
    train_feat = os.path.join(args.data_dir, args.train_split, "features.pkl")
    train_lab = os.path.join(args.data_dir, args.train_split, "labels.pkl")
    dev_feat = os.path.join(args.data_dir, args.dev_split, "features.pkl")
    dev_lab = os.path.join(args.data_dir, args.dev_split, "labels.pkl")
    test_feat = os.path.join(args.data_dir, args.test_split, "features.pkl")
    test_lab = os.path.join(args.data_dir, args.test_split, "labels.pkl")  # may not exist

    # Load train/dev datasets if training
    if args.epochs > 0:
        train_ds = AudioDataset(train_feat, train_lab)
        dev_ds = AudioDataset(dev_feat, dev_lab)

        # Infer feature dim (C)
        _, feat0, _ = train_ds[0]
        in_ch = int(feat0.shape[0])

        # Weighted sampler for class imbalance
        y_train = np.array([int(train_ds.label_map[u]) for u in train_ds.features_df["uttid"].tolist()], dtype=int)
        pos_weight, w0, w1 = compute_class_weights(y_train)
        sample_weights = np.array([w1 if y == 1 else w0 for y in y_train], dtype=np.float32)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=False
        )
        dev_loader = DataLoader(
            dev_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=False
        )

        model = DeepfakeDetector(in_ch=in_ch, hidden=args.hidden, dropout=args.dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

        ema = EMA(model, decay=args.ema_decay) if args.ema else None

        best_eer = 1.0
        best_path = args.ckpt_path
        bad = 0

        for epoch in range(1, args.epochs + 1):
            model.train()
            pbar = tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}", leave=False)
            total_loss = 0.0

            for uttids, x, lengths, y in pbar:
                x = x.to(device)
                lengths = lengths.to(device)
                y = y.to(device)

                # Per-sample SpecAugment (only training)
                if args.specaug:
                    # apply on CPU-ish tensors? We are already on GPU. We'll do it on GPU.
                    # x: (B,C,T)
                    for i in range(x.size(0)):
                        xi = x[i]
                        xi = time_mask(xi, args.time_mask_max, args.time_mask_n)
                        xi = freq_mask(xi, args.freq_mask_max, args.freq_mask_n)
                        x[i] = xi

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    logits = model(x, lengths)
                    loss = criterion(logits, y)

                scaler.scale(loss).backward()
                if args.grad_clip and args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()

                if ema is not None:
                    ema.update(model)

                total_loss += float(loss.item())
                pbar.set_postfix(loss=total_loss / max(1, len(pbar)))

            # Evaluate (EMA weights if enabled)
            if ema is not None:
                ema.apply_to(model)
            dev_eer = evaluate_eer(model, dev_loader, device=device)
            if ema is not None:
                ema.restore(model)

            print(f"Epoch {epoch}: dev EER={dev_eer:.6f}  (lower is better)")

            if dev_eer < best_eer:
                best_eer = dev_eer
                bad = 0
                # save best model weights
                torch.save(model.state_dict(), best_path)
                print(f"  ✓ Saved best ckpt -> {best_path}")
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"Early stopping: no improvement for {args.patience} epochs.")
                    break

        print(f"\nTraining done. Best dev EER: {best_eer:.6f}")
    else:
        # Inference-only: need to infer in_ch from test features
        test_ds_tmp = AudioDataset(test_feat, labels_pkl=None)
        _, feat0, _ = test_ds_tmp[0]
        in_ch = int(feat0.shape[0])
        model = DeepfakeDetector(in_ch=in_ch, hidden=args.hidden, dropout=args.dropout).to(device)

    # Load checkpoint (best/dev-selected)
    if os.path.exists(args.ckpt_path):
        state = torch.load(args.ckpt_path, map_location=device)
        model.load_state_dict(state)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")

    # Inference on requested split
    test_ds = AudioDataset(test_feat, labels_pkl=test_lab if os.path.exists(test_lab) else None)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=False
    )

    pred_df = run_inference(model, test_loader, device=device, use_prob=args.use_prob)
    pred_df.to_pickle(args.prediction_pkl)
    print(f"Saved prediction file -> {args.prediction_pkl}")
    print(pred_df.head())
    print("shape:", pred_df.shape)

    # If labels exist for this split, also compute EER (useful for test1/dev sanity checks)
    if test_ds.has_labels:
        # Reuse the same loader but y is available
        eer = evaluate_eer(model, test_loader, device=device)
        print(f"\nEER on split '{args.test_split}': {eer:.6f}")
    else:
        print("\nDone! (Inference only on unlabeled split, no EER computed)")

if __name__ == "__main__":
    main()
