"""Compare Raw vs CMN vs CVMN preprocessing on the same CNN2D architecture.

Trains 3 models (30 epochs each, same seed) and reports Dev/Test1 EER.
Run from project root:
    conda activate intro2dl && python src/compare_normalization.py
"""

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from model import CNN2D
from evaluation import calculate_eer

try:
    from rich.console import Console
    from rich.live import Live
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


class NormalizedDataset(Dataset):
    """Dataset with optional per-utterance CMN or CVMN."""

    def __init__(self, features_path, labels_path, mode="raw"):
        features_df = pd.read_pickle(features_path)
        labels_df = pd.read_pickle(labels_path)
        merged = pd.merge(features_df, labels_df, on="uttid", how="inner")
        merged = merged.reset_index(drop=True)
        self.features = merged["features"].tolist()
        self.labels = merged["label"].tolist()
        self.mode = mode  # "raw", "cmn", "cvmn"

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx].float()  # stored as (180, 321)

        if self.mode == "cmn":
            # Subtract per-feature mean across time (dim=1 is time for stored shape)
            feat = feat - feat.mean(dim=1, keepdim=True)
        elif self.mode == "cvmn":
            mean = feat.mean(dim=1, keepdim=True)
            std = feat.std(dim=1, keepdim=True).clamp(min=1e-8)
            feat = (feat - mean) / std

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return feat, label


def make_criterion():
    bce = nn.BCEWithLogitsLoss()
    eps = LABEL_SMOOTHING

    def criterion(logits, y):
        y_smooth = y * (1.0 - eps) + 0.5 * eps
        return bce(logits, y_smooth)

    return criterion


def train_and_score(mode, console=None):
    """Train CNN2D with given normalization mode, return (dev_eer, test1_eer, score_mean, score_std)."""
    torch.manual_seed(SEED)

    train_loader = DataLoader(
        NormalizedDataset("data/train/features.pkl", "data/train/labels.pkl", mode),
        batch_size=32, shuffle=True, num_workers=2,
    )
    dev_loader = DataLoader(
        NormalizedDataset("data/dev/features.pkl", "data/dev/labels.pkl", mode),
        batch_size=32, shuffle=False,
    )
    test1_loader = DataLoader(
        NormalizedDataset("data/test1/features.pkl", "data/test1/test1_labels.pkl", mode),
        batch_size=32, shuffle=False,
    )

    model = CNN2D(in_features=180, dropout=0.2).to(DEVICE)
    criterion = make_criterion()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6,
    )

    best_eer = float("inf")
    best_state = None
    no_improve = 0

    epoch_progress = None
    if console:
        epoch_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        )

    ctx = epoch_progress if epoch_progress else None
    task_id = None

    if ctx:
        ctx.start()
        task_id = ctx.add_task(f"{mode.upper():>6s}", total=EPOCHS)

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        for features, labels in train_loader:
            features = features.to(DEVICE).transpose(1, 2)
            labels = labels.to(DEVICE)
            loss = criterion(model(features).squeeze(-1), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Dev eval
        model.eval()
        scores, lbls = [], []
        with torch.no_grad():
            for features, labels in dev_loader:
                logits = model(features.to(DEVICE).transpose(1, 2)).squeeze(-1)
                scores.extend(torch.sigmoid(logits).cpu().tolist())
                lbls.extend(labels.tolist())

        dev_eer, _ = calculate_eer(scores, lbls)
        scheduler.step(dev_eer)

        if ctx and task_id is not None:
            ctx.update(task_id, advance=1,
                       description=f"{mode.upper():>6s} ep{epoch:2d} dev_eer={dev_eer:.4f}")

        if dev_eer < best_eer:
            best_eer = dev_eer
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= EARLY_STOP:
            if ctx and task_id is not None:
                ctx.update(task_id, completed=EPOCHS,
                           description=f"{mode.upper():>6s} early stop ep{epoch} dev_eer={best_eer:.4f}")
            break

    if ctx:
        ctx.stop()

    # Test1 eval with best model
    model.load_state_dict(best_state)
    model.to(DEVICE).eval()
    scores, lbls = [], []
    with torch.no_grad():
        for features, labels in test1_loader:
            logits = model(features.to(DEVICE).transpose(1, 2)).squeeze(-1)
            scores.extend(torch.sigmoid(logits).cpu().tolist())
            lbls.extend(labels.tolist())

    test1_eer, _ = calculate_eer(scores, lbls)
    s = np.array(scores)
    return best_eer, test1_eer, s.mean(), s.std()


def main():
    console = Console() if HAS_RICH else None

    if console:
        console.print(Panel(
            f"Device: {DEVICE}  |  Seed: {SEED}  |  Epochs: {EPOCHS}  |  "
            f"Early stop: {EARLY_STOP}\n"
            f"Label smoothing: {LABEL_SMOOTHING}  |  "
            f"Architecture: CNN2D (same as submitted model)",
            title="CMN/CVMN Comparison Experiment", border_style="blue",
        ))

    results = {}
    for mode in ["raw", "cmn", "cvmn"]:
        if console:
            console.print(f"\n[bold cyan]Training with {mode.upper()}...[/]")
        else:
            print(f"\nTraining with {mode.upper()}...")

        results[mode] = train_and_score(mode, console=console)
        dev_eer, t1_eer, smean, sstd = results[mode]

        if console:
            console.print(f"  [green]Done.[/] dev_eer={dev_eer:.4%}  test1_eer={t1_eer:.4%}")
        else:
            print(f"  Done. dev_eer={dev_eer:.4%}  test1_eer={t1_eer:.4%}")

    # Summary table
    if console:
        table = Table(title="Normalization Comparison Results")
        table.add_column("Mode", style="cyan", width=8)
        table.add_column("Dev EER", justify="right", width=12)
        table.add_column("Test1 EER", justify="right", width=12)
        table.add_column("Score Mean", justify="right", width=12)
        table.add_column("Score Std", justify="right", width=12)

        for mode in ["raw", "cmn", "cvmn"]:
            dev_eer, t1_eer, smean, sstd = results[mode]
            table.add_row(
                mode.upper(),
                f"{dev_eer:.4%}",
                f"{t1_eer:.4%}",
                f"{smean:.4f}",
                f"{sstd:.4f}",
            )

        console.print()
        console.print(table)
        console.print(Panel(
            "CMN = per-utterance mean subtraction (removes channel bias)\n"
            "CVMN = per-utterance mean+variance normalization (removes channel bias + scale)\n"
            "Raw = no per-utterance normalization (what we submitted)",
            title="What These Mean", border_style="dim",
        ))
    else:
        print(f"\n{'='*60}")
        print(f"{'Mode':>8s}  {'Dev EER':>10s}  {'Test1 EER':>10s}  {'Score Mean':>10s}  {'Score Std':>10s}")
        print(f"{'-'*60}")
        for mode in ["raw", "cmn", "cvmn"]:
            dev_eer, t1_eer, smean, sstd = results[mode]
            print(f"{mode.upper():>8s}  {dev_eer:10.4%}  {t1_eer:10.4%}  {smean:10.4f}  {sstd:10.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
