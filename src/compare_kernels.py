"""Compare CNN1D kernel configurations: (3,3,3) vs (5,3,3).

Trains models, saves best checkpoints, reports Dev/Test1 EER.
Run from project root:
    conda activate intro2dl && python src/compare_kernels.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from evaluation import calculate_eer

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn, MofNCompleteColumn, Progress,
        SpinnerColumn, TextColumn, TimeElapsedColumn,
    )
    from rich.table import Table
    from rich.panel import Panel
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 42
EPOCHS = 30
EARLY_STOP = 8
LABEL_SMOOTHING = 0.05
CKPT_DIR = "checkpoints/kernel_comparison"


class CNN1D_Variant(nn.Module):
    """CNN1D with configurable kernel sizes per layer."""

    def __init__(self, in_features=180, base_channels=32, num_classes=1,
                 dropout=0.2, kernels=(3, 3, 3)):
        super().__init__()
        k1, k2, k3 = kernels
        p1, p2, p3 = k1 // 2, k2 // 2, k3 // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_features, base_channels, kernel_size=k1, padding=p1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=k2, padding=p2),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=k3, padding=p3),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(base_channels * 4, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)


class NormalizedDataset(Dataset):
    def __init__(self, features_path, labels_path, mode="raw"):
        features_df = pd.read_pickle(features_path)
        labels_df = pd.read_pickle(labels_path)
        merged = pd.merge(features_df, labels_df, on="uttid", how="inner")
        merged = merged.reset_index(drop=True)
        self.features = merged["features"].tolist()
        self.labels = merged["label"].tolist()
        self.mode = mode

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx].float()
        if self.mode == "cmn":
            feat = feat - feat.mean(dim=1, keepdim=True)
        elif self.mode == "cvmn":
            mean = feat.mean(dim=1, keepdim=True)
            std = feat.std(dim=1, keepdim=True).clamp(min=1e-8)
            feat = (feat - mean) / std
        return feat, torch.tensor(self.labels[idx], dtype=torch.float32)


def infer_scores(model, loader):
    model.train(False)
    scores, lbls = [], []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(DEVICE).transpose(1, 2)
            logits = model(features).squeeze(-1)
            scores.extend(torch.sigmoid(logits).cpu().tolist())
            lbls.extend(labels.tolist())
    return scores, lbls


def train_and_score(kernels, norm_mode, save_tag, console=None):
    torch.manual_seed(SEED)
    label = f"k={','.join(map(str, kernels))} {norm_mode}"

    train_loader = DataLoader(
        NormalizedDataset("data/train/features.pkl", "data/train/labels.pkl", norm_mode),
        batch_size=32, shuffle=True, num_workers=2)
    dev_loader = DataLoader(
        NormalizedDataset("data/dev/features.pkl", "data/dev/labels.pkl", norm_mode),
        batch_size=32, shuffle=False)
    test1_loader = DataLoader(
        NormalizedDataset("data/test1/features.pkl", "data/test1/test1_labels.pkl", norm_mode),
        batch_size=32, shuffle=False)

    model = CNN1D_Variant(in_features=180, dropout=0.2, kernels=kernels).to(DEVICE)
    eps = LABEL_SMOOTHING
    bce = nn.BCEWithLogitsLoss()
    def criterion(logits, y):
        return bce(logits, y * (1.0 - eps) + 0.5 * eps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)

    best_eer, best_state, best_epoch, no_improve = float("inf"), None, 0, 0

    progress = None
    task_id = None
    if console:
        progress = Progress(
            SpinnerColumn(), TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=30), MofNCompleteColumn(), TimeElapsedColumn(),
            console=console)
        progress.start()
        task_id = progress.add_task(label, total=EPOCHS)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for features, labels in train_loader:
            features = features.to(DEVICE).transpose(1, 2)
            labels = labels.to(DEVICE)
            loss = criterion(model(features).squeeze(-1), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scores, lbls = infer_scores(model, dev_loader)
        dev_eer, _ = calculate_eer(scores, lbls)
        scheduler.step(dev_eer)

        if progress and task_id is not None:
            progress.update(task_id, advance=1,
                            description=f"{label} ep{epoch:2d} eer={dev_eer:.4f}")
        if dev_eer < best_eer:
            best_eer = dev_eer
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= EARLY_STOP:
            if progress and task_id is not None:
                progress.update(task_id, completed=EPOCHS,
                                description=f"{label} STOP ep{epoch} eer={best_eer:.4f}")
            break

    if progress:
        progress.stop()

    # Save best checkpoint
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f"{save_tag}_best.pt")
    torch.save({
        "model_state": best_state,
        "kernels": kernels,
        "norm_mode": norm_mode,
        "best_epoch": best_epoch,
        "best_dev_eer": best_eer,
    }, ckpt_path)

    # Test1 with best weights
    model.load_state_dict(best_state)
    model.to(DEVICE)
    scores, lbls = infer_scores(model, test1_loader)
    test1_eer, _ = calculate_eer(scores, lbls)
    s = np.array(scores)

    return best_eer, test1_eer, s.mean(), s.std(), ckpt_path


def main():
    console = Console() if HAS_RICH else None

    experiments = [
        ((3, 3, 3), "raw",  "cnn1d_k333_raw",  "CNN1D k=3,3,3 raw (baseline)"),
        ((5, 3, 3), "raw",  "cnn1d_k533_raw",  "CNN1D k=5,3,3 raw"),
        ((5, 3, 3), "cmn",  "cnn1d_k533_cmn",  "CNN1D k=5,3,3 + CMN"),
        ((5, 3, 3), "cvmn", "cnn1d_k533_cvmn", "CNN1D k=5,3,3 + CVMN"),
    ]

    if console:
        console.print(Panel(
            f"Device: {DEVICE}  |  Seed: {SEED}  |  Epochs: {EPOCHS}  |  "
            f"Early stop: {EARLY_STOP}\n"
            f"Base architecture: CNN1D (180->32->64->128), mean pool\n"
            f"Testing: kernel width + CMN/CVMN normalization\n"
            f"Checkpoints saved to: {CKPT_DIR}/",
            title="Kernel Size Comparison Experiment", border_style="blue"))

    results = []
    for kernels, norm, tag, name in experiments:
        if console:
            console.print(f"\n[bold cyan]{name}[/]")
        else:
            print(f"\n{name}")

        dev_eer, t1_eer, smean, sstd, ckpt = train_and_score(
            kernels, norm, tag, console)
        results.append((name, dev_eer, t1_eer, smean, sstd, ckpt))

        msg = f"dev={dev_eer:.4%}  test1={t1_eer:.4%}  ckpt={ckpt}"
        if console:
            console.print(f"  [green]Done.[/] {msg}")
        else:
            print(f"  Done. {msg}")

    if console:
        table = Table(title="Kernel Size Comparison Results")
        table.add_column("Experiment", width=30)
        table.add_column("Dev EER", justify="right", width=10)
        table.add_column("Test1 EER", justify="right", width=10)
        table.add_column("Score Mean", justify="right", width=10)
        table.add_column("Score Std", justify="right", width=10)
        table.add_column("Checkpoint", width=35)
        for name, dev_eer, t1_eer, smean, sstd, ckpt in results:
            table.add_row(name, f"{dev_eer:.4%}", f"{t1_eer:.4%}",
                          f"{smean:.4f}", f"{sstd:.4f}", ckpt)
        console.print()
        console.print(table)
    else:
        print(f"\n{'='*80}")
        for name, dev_eer, t1_eer, smean, sstd, ckpt in results:
            print(f"{name:>32s}  dev={dev_eer:.4%}  t1={t1_eer:.4%}  {ckpt}")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
