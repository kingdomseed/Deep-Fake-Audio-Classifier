import argparse
import csv
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from augmentation import spec_augment
from dataloaders import make_loader
from evaluation import evaluate
from model import build_model
from training import save_checkpoint
from visualizers import BatchMetrics, EpochMetrics, TrainingConfig, create_visualizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multiple model trainings and rank by best dev EER."
    )
    parser.add_argument("--train-features", default="data/train/features.pkl")
    parser.add_argument("--train-labels", default="data/train/labels.pkl")
    parser.add_argument("--dev-features", default="data/dev/features.pkl")
    parser.add_argument("--dev-labels", default="data/dev/labels.pkl")
    parser.add_argument("--models", default="cnn2d,cnn2d_robust")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--early-stop", type=int, default=3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--in-features", type=int, default=321)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout for CNN models.",
    )
    parser.add_argument("--pool-bins", type=int, default=1)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--seeds", default="0", help="comma-separated list of seeds")
    parser.add_argument(
        "--spec-augment",
        action="store_true",
        help="enable SpecAugment for all models (time masking, optionally feature masking)",
    )
    parser.add_argument(
        "--time-mask-ratio",
        type=float,
        default=0.2,
        help="max ratio of time steps to mask (default: 0.2)",
    )
    parser.add_argument(
        "--feature-mask-ratio",
        type=float,
        default=0.1,
        help="max ratio of features to mask (default: 0.1)",
    )
    parser.add_argument(
        "--feature-mask",
        action="store_true",
        help="enable feature masking in addition to time masking",
    )
    parser.add_argument(
        "--visualizer",
        default="noop",
        choices=["rich", "tqdm", "noop"],
        help="training visualizer per model",
    )
    return parser.parse_args()


def resolve_device(device_arg: Optional[str]) -> str:
    if device_arg:
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_seeds(seed_str: str) -> List[int]:
    seeds = [s.strip() for s in seed_str.split(",") if s.strip()]
    return [int(s) for s in seeds] if seeds else [0]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model_kwargs(model_name: str, args) -> Dict:
    if model_name in {"cnn1d"}:
        dropout = args.dropout
        return {
            "in_channels": args.in_features,
            "dropout": dropout,
            "pool_bins": args.pool_bins,
        }
    if model_name in {"cnn2d", "cnn2d_robust"}:
        dropout = args.dropout
        return {
            "in_features": args.in_features,
            "dropout": dropout,
        }
    if model_name in {"crnn", "crnn2"}:
        dropout = args.dropout
        return {
            "in_features": args.in_features,
            "dropout": dropout,
        }
    return {}


def parse_model_spec(spec: str) -> Tuple[str, str, bool]:
    """
    Parse model spec allowing per-model SpecAugment flag.
    Use suffix "+specaug" to enable augmentation for a specific model.
    Example: cnn2d+specaug
    """
    spec = spec.strip()
    if spec.endswith("+specaug"):
        base = spec[: -len("+specaug")]
        return spec, base, True
    return spec, spec, False


def build_augment_fn(args, enabled: bool):
    if not enabled:
        return None

    def augment_fn(features):
        return spec_augment(
            features,
            time_mask_ratio=args.time_mask_ratio,
            feature_mask_ratio=args.feature_mask_ratio,
            apply_time_mask=True,
            apply_feature_mask=args.feature_mask,
        )

    return augment_fn


def train_one_model(
    model_name: str,
    base_model: str,
    args,
    device: str,
    seed: int,
    use_specaugment: bool,
) -> Dict:
    train_loader = make_loader(
        args.train_features,
        args.train_labels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    dev_loader = make_loader(
        args.dev_features,
        args.dev_labels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    model_kwargs = build_model_kwargs(base_model, args)
    model = build_model(base_model, **model_kwargs).to(device)

    criterion = nn.BCEWithLogitsLoss()
    use_adamw = base_model.startswith("cnn")
    weight_decay = args.weight_decay
    if use_adamw and weight_decay == 0.0:
        weight_decay = 0.01
    if weight_decay > 0:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    augment_fn = build_augment_fn(args, args.spec_augment or use_specaugment)
    if augment_fn is not None:
        feature_mask_status = (
            f"{args.feature_mask_ratio:.2f}" if args.feature_mask else "disabled"
        )
        print(
            f"SpecAugment enabled for {model_name}: "
            f"time_mask={args.time_mask_ratio:.2f}, "
            f"feature_mask={feature_mask_status}"
        )

    visualizer = create_visualizer(args.visualizer)
    config = TrainingConfig(
        device=device,
        model=model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=weight_decay,
        early_stop_patience=args.early_stop,
        in_features=args.in_features,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    visualizer.on_training_start(config)

    best_eer = None
    best_metrics: Optional[EpochMetrics] = None
    prev_metrics: Optional[EpochMetrics] = None
    epochs_no_improve = 0
    history: List[EpochMetrics] = []

    for epoch in range(1, args.epochs + 1):
        with visualizer.on_epoch_start(epoch, len(train_loader)) as batch_ctx:
            train_loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                batch_ctx,
                augment_fn=augment_fn,
            )

        dev_metrics, _, _ = evaluate(
            model, dev_loader, criterion=criterion, device=device
        )

        is_best = False
        if dev_metrics["eer"] is not None:
            if best_eer is None or dev_metrics["eer"] < best_eer:
                is_best = True
                best_eer = dev_metrics["eer"]
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

        improved = (
            prev_metrics is not None
            and prev_metrics.dev_eer is not None
            and dev_metrics["eer"] is not None
            and dev_metrics["eer"] < prev_metrics.dev_eer
        )

        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            dev_loss=dev_metrics["avg_loss"],
            dev_eer=dev_metrics["eer"],
            is_best=is_best,
            improved=improved,
            epochs_no_improve=epochs_no_improve,
        )
        visualizer.on_epoch_end(metrics, prev_metrics)
        history.append(metrics)
        prev_metrics = metrics

        if is_best:
            best_metrics = metrics
            save_checkpoint(
                model,
                optimizer,
                epoch,
                args,
                os.path.join(args.checkpoint_dir, f"{model_name}_best.pt"),
            )

        if args.early_stop and epochs_no_improve >= args.early_stop:
            print(
                f"\nEarly stopping {model_name} at epoch {epoch} "
                f"(no improvement in {args.early_stop} epochs)"
            )
            break

    visualizer.on_training_end(history)
    save_checkpoint(
        model,
        optimizer,
        history[-1].epoch if history else args.epochs,
        args,
        os.path.join(args.checkpoint_dir, f"{model_name}_last.pt"),
    )

    return {
        "model": model_name,
        "seed": seed,
        "best_eer": best_metrics.dev_eer if best_metrics else None,
        "best_epoch": best_metrics.epoch if best_metrics else None,
        "best_dev_loss": best_metrics.dev_loss if best_metrics else None,
        "history": history,
    }


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    batch_context=None,
    augment_fn=None,
):
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch_idx, (features, labels) in enumerate(dataloader):
        features = features.to(device)
        labels = labels.to(device)

        if augment_fn is not None:
            features = augment_fn(features)

        logits = model(features).squeeze(-1)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_count += labels.size(0)

        if batch_context is not None and total_count > 0:
            batch_context.update_batch(
                BatchMetrics(
                    batch_idx=batch_idx,
                    running_loss=total_loss / total_count,
                    batch_size=labels.size(0),
                )
            )

    return (total_loss / total_count) if total_count > 0 else None


def save_results_csv(results: List[Dict], path: str, fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def save_results_plot(results: List[Dict], path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping plot generation.")
        return

    models = [r["model"] for r in results]
    eers = [r["mean_eer"] if r["mean_eer"] is not None else 1.0 for r in results]
    yerr = [r["std_eer"] if r["std_eer"] is not None else 0.0 for r in results]

    plt.figure(figsize=(8, 4))
    plt.bar(models, eers, color="#4c78a8", yerr=yerr, capsize=4)
    plt.title("Best Dev EER by Model (lower is better)")
    plt.ylabel("Best Dev EER")
    plt.tight_layout()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def flatten_history(model_name: str, seed: int, history: List[EpochMetrics]) -> List[Dict]:
    rows = []
    for m in history:
        rows.append(
            {
                "model": model_name,
                "seed": seed,
                "epoch": m.epoch,
                "train_loss": m.train_loss,
                "dev_loss": m.dev_loss,
                "dev_eer": m.dev_eer,
                "is_best": m.is_best,
            }
        )
    return rows


def avg_losses_up_to_best(model_epoch_rows: List[Dict], seed: int, model_name: str):
    df = [r for r in model_epoch_rows if r["seed"] == seed and r["model"] == model_name]
    if not df:
        return None, None, None
    # find best epoch by dev_eer
    best_row = min(df, key=lambda r: float("inf") if r["dev_eer"] is None else r["dev_eer"])
    best_epoch = best_row["epoch"]
    upto = [r for r in df if r["epoch"] <= best_epoch]
    if not upto:
        return best_epoch, None, None
    train_vals = [r["train_loss"] for r in upto if r["train_loss"] is not None]
    dev_vals = [r["dev_loss"] for r in upto if r["dev_loss"] is not None]
    avg_train = float(np.mean(train_vals)) if train_vals else None
    avg_dev = float(np.mean(dev_vals)) if dev_vals else None
    return best_epoch, avg_train, avg_dev


def aggregate_history(rows: List[Dict]) -> Dict[int, Dict[str, Optional[float]]]:
    by_epoch: Dict[int, Dict[str, List[float]]] = {}
    for r in rows:
        epoch = int(r["epoch"])
        by_epoch.setdefault(epoch, {"train_loss": [], "dev_loss": [], "dev_eer": []})
        if r["train_loss"] is not None:
            by_epoch[epoch]["train_loss"].append(float(r["train_loss"]))
        if r["dev_loss"] is not None:
            by_epoch[epoch]["dev_loss"].append(float(r["dev_loss"]))
        if r["dev_eer"] is not None:
            by_epoch[epoch]["dev_eer"].append(float(r["dev_eer"]))

    stats: Dict[int, Dict[str, Optional[float]]] = {}
    for epoch, vals in by_epoch.items():
        stats[epoch] = {
            "train_loss_mean": float(np.mean(vals["train_loss"])) if vals["train_loss"] else None,
            "train_loss_std": float(np.std(vals["train_loss"])) if len(vals["train_loss"]) > 1 else 0.0 if vals["train_loss"] else None,
            "dev_loss_mean": float(np.mean(vals["dev_loss"])) if vals["dev_loss"] else None,
            "dev_loss_std": float(np.std(vals["dev_loss"])) if len(vals["dev_loss"]) > 1 else 0.0 if vals["dev_loss"] else None,
            "dev_eer_mean": float(np.mean(vals["dev_eer"])) if vals["dev_eer"] else None,
            "dev_eer_std": float(np.std(vals["dev_eer"])) if len(vals["dev_eer"]) > 1 else 0.0 if vals["dev_eer"] else None,
        }
    return stats


def estimate_overfit_epoch(stats: Dict[int, Dict[str, float]]) -> Optional[int]:
    epochs = sorted(stats.keys())
    if len(epochs) < 4:
        return None
    for i in range(2, len(epochs)):
        e0, e1, e2 = epochs[i - 2], epochs[i - 1], epochs[i]
        t0 = stats[e0]["train_loss_mean"]
        t1 = stats[e1]["train_loss_mean"]
        t2 = stats[e2]["train_loss_mean"]
        d0 = stats[e0]["dev_loss_mean"]
        d1 = stats[e1]["dev_loss_mean"]
        d2 = stats[e2]["dev_loss_mean"]
        if None in (t0, t1, t2, d0, d1, d2):
            continue
        train_down = (t2 < t1) and (t1 <= t0)
        dev_up = (d2 > d1) and (d1 >= d0)
        if train_down and dev_up:
            return e2
    return None


def save_model_curves_plot(
    model_name: str,
    stats: Dict[int, Dict[str, float]],
    path: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping plot generation.")
        return

    epochs = sorted(stats.keys())
    def to_float_list(key: str):
        return [stats[e][key] if stats[e][key] is not None else np.nan for e in epochs]

    train_mean = to_float_list("train_loss_mean")
    train_std = to_float_list("train_loss_std")
    dev_mean = to_float_list("dev_loss_mean")
    dev_std = to_float_list("dev_loss_std")
    eer_mean = to_float_list("dev_eer_mean")
    eer_std = to_float_list("dev_eer_std")

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_mean, label="train loss", color="#4c78a8")
    plt.plot(epochs, dev_mean, label="dev loss", color="#f58518")
    train_mean_arr = np.array(train_mean, dtype=float)
    train_std_arr = np.array(train_std, dtype=float)
    dev_mean_arr = np.array(dev_mean, dtype=float)
    dev_std_arr = np.array(dev_std, dtype=float)
    eer_mean_arr = np.array(eer_mean, dtype=float)
    eer_std_arr = np.array(eer_std, dtype=float)

    if not np.all(np.isnan(train_std_arr)) and np.nanmax(train_std_arr) > 0:
        plt.fill_between(epochs, train_mean_arr - train_std_arr, train_mean_arr + train_std_arr, alpha=0.15, color="#4c78a8")
    if not np.all(np.isnan(dev_std_arr)) and np.nanmax(dev_std_arr) > 0:
        plt.fill_between(epochs, dev_mean_arr - dev_std_arr, dev_mean_arr + dev_std_arr, alpha=0.15, color="#f58518")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, eer_mean, label="dev EER", color="#54a24b")
    if not np.all(np.isnan(eer_std_arr)) and np.nanmax(eer_std_arr) > 0:
        plt.fill_between(epochs, eer_mean_arr - eer_std_arr, eer_mean_arr + eer_std_arr, alpha=0.15, color="#54a24b")
    plt.xlabel("Epoch")
    plt.ylabel("EER")
    plt.legend()

    plt.suptitle(f"{model_name} - Training Curves")
    plt.tight_layout()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def write_report(
    path: str,
    summary_results: List[Dict],
    per_model_stats: Dict[str, Dict[int, Dict[str, float]]],
    overfit_epochs: Dict[str, Optional[int]],
    plot_paths: Dict[str, str],
    args,
) -> None:
    lines = []
    lines.append("# Model Comparison Report")
    lines.append("")
    lines.append("## Experiment Setup")
    lines.append(f"- Epochs: {args.epochs}")
    lines.append(f"- Batch size: {args.batch_size}")
    lines.append(f"- Learning rate: {args.lr}")
    lines.append(f"- Dropout (CNNs): {args.dropout}")
    lines.append(f"- Pool bins (CNN1D): {args.pool_bins}")
    lines.append(f"- Seeds: {args.seeds}")
    lines.append("- Optimizer policy:")
    lines.append("  - CNNs: AdamW with weight decay 0.01 by default (unless --weight-decay overrides)")
    lines.append("  - MLPs: Adam unless --weight-decay > 0")
    lines.append("")
    lines.append("## Summary Table (mean EER, lower is better)")
    lines.append("")
    lines.append("| Model | Mean EER | Std | Best EER | Best Epoch | Best Seed | Avg Train Loss (<= best) | Avg Dev Loss (<= best) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in summary_results:
        mean_eer = f"{r['mean_eer']:.4f}" if r["mean_eer"] is not None else "N/A"
        std_eer = f"{r['std_eer']:.4f}" if r["std_eer"] is not None else "N/A"
        best_eer = f"{r['best_eer']:.4f}" if r["best_eer"] is not None else "N/A"
        avg_train = f"{r['avg_train_loss_upto_best']:.4f}" if r.get("avg_train_loss_upto_best") is not None else "N/A"
        avg_dev = f"{r['avg_dev_loss_upto_best']:.4f}" if r.get("avg_dev_loss_upto_best") is not None else "N/A"
        lines.append(
            f"| {r['model']} | {mean_eer} | {std_eer} | {best_eer} | {r['best_epoch']} | {r['best_seed']} | {avg_train} | {avg_dev} |"
        )
    lines.append("")
    lines.append("## Overfitting Signals (heuristic)")
    lines.append("First epoch where average train loss keeps decreasing while average dev loss rises for two consecutive steps.")
    lines.append("")
    for model, epoch in overfit_epochs.items():
        if epoch is None:
            lines.append(f"- {model}: no clear overfitting signal in averaged curves")
        else:
            lines.append(f"- {model}: potential overfitting starts around epoch {epoch}")
    lines.append("")
    lines.append("## Plots")
    if "combined" in plot_paths:
        rel_path = plot_paths["combined"].replace(os.getcwd() + os.sep, "")
        lines.append(f"- combined: `{rel_path}`")
    for model, plot_path in plot_paths.items():
        if model == "combined":
            continue
        rel_path = plot_path.replace(os.getcwd() + os.sep, "")
        lines.append(f"- {model}: `{rel_path}`")
    lines.append("")
    lines.append("## Notes / Justifications")
    lines.append("- Mean pooling vs max pooling: mean pooling preserves the average energy pattern across time and is less sensitive to single-frame outliers; max pooling can over-emphasize spiky artifacts and destabilize training for this dataset.")
    lines.append("- CNNs preserve local temporal patterns that MLP pooling removes; this typically improves EER when deepfake artifacts are short and localized.")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))


def save_combined_loss_plot(
    per_model_stats: Dict[str, Dict[int, Dict[str, float]]],
    path: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping combined plot generation.")
        return

    plt.figure(figsize=(10, 6))
    for model_name, stats in per_model_stats.items():
        epochs = sorted(stats.keys())
        train_mean = [
            stats[e]["train_loss_mean"] if stats[e]["train_loss_mean"] is not None else np.nan
            for e in epochs
        ]
        dev_mean = [
            stats[e]["dev_loss_mean"] if stats[e]["dev_loss_mean"] is not None else np.nan
            for e in epochs
        ]
        plt.plot(epochs, train_mean, label=f"{model_name} train")
        plt.plot(epochs, dev_mean, linestyle="--", label=f"{model_name} dev")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Dev Loss (All Models)")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    device = resolve_device(args.device)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = parse_seeds(args.seeds)

    run_results = []
    summary_results = []
    epoch_rows = []
    per_model_stats = {}
    overfit_epochs = {}
    plot_paths = {}

    for model_spec in models:
        model_name, base_model, use_specaugment = parse_model_spec(model_spec)
        model_runs = []
        for seed in seeds:
            print(f"\n=== Training {model_name} (seed {seed}) ===")
            set_seed(seed)
            result = train_one_model(
                model_name,
                base_model,
                args,
                device,
                seed,
                use_specaugment,
            )
            run_results.append(
                {
                    "model": result["model"],
                    "seed": result["seed"],
                    "best_eer": result["best_eer"],
                    "best_epoch": result["best_epoch"],
                    "best_dev_loss": result["best_dev_loss"],
                }
            )
            model_runs.append(result)
            epoch_rows.extend(flatten_history(model_name, seed, result["history"]))

        eers = [r["best_eer"] for r in model_runs if r["best_eer"] is not None]
        mean_eer = float(np.mean(eers)) if eers else None
        if len(eers) > 1:
            std_eer = float(np.std(eers))
        elif len(eers) == 1:
            std_eer = 0.0
        else:
            std_eer = None
        best_run = min(model_runs, key=lambda r: float("inf") if r["best_eer"] is None else r["best_eer"])

        # compute averages up to best epoch for the best seed
        best_seed = best_run["seed"]
        best_epoch, avg_train_upto_best, avg_dev_upto_best = avg_losses_up_to_best(
            epoch_rows, best_seed, model_name
        )

        summary_results.append(
            {
                "model": model_name,
                "mean_eer": mean_eer,
                "std_eer": std_eer,
                "best_eer": best_run["best_eer"],
                "best_epoch": best_run["best_epoch"],
                "best_seed": best_seed,
                "avg_train_loss_upto_best": avg_train_upto_best,
                "avg_dev_loss_upto_best": avg_dev_upto_best,
                "runs": len(model_runs),
            }
        )

    summary_results.sort(key=lambda r: (r["mean_eer"] is None, r["mean_eer"]))

    runs_csv_path = os.path.join(args.results_dir, "model_runs.csv")
    epochs_csv_path = os.path.join(args.results_dir, "model_epochs.csv")
    ranking_csv_path = os.path.join(args.results_dir, "model_ranking.csv")
    plot_path = os.path.join(args.results_dir, "model_ranking.png")

    save_results_csv(
        run_results,
        runs_csv_path,
        fieldnames=["model", "seed", "best_eer", "best_epoch", "best_dev_loss"],
    )
    save_results_csv(
        epoch_rows,
        epochs_csv_path,
        fieldnames=["model", "seed", "epoch", "train_loss", "dev_loss", "dev_eer", "is_best"],
    )
    save_results_csv(
        summary_results,
        ranking_csv_path,
        fieldnames=[
            "model",
            "mean_eer",
            "std_eer",
            "best_eer",
            "best_epoch",
            "best_seed",
            "avg_train_loss_upto_best",
            "avg_dev_loss_upto_best",
            "runs",
        ],
    )
    save_results_plot(summary_results, plot_path)

    # Per-model curves + overfitting heuristics
    for model_name in models:
        model_epoch_rows = [r for r in epoch_rows if r["model"] == model_name]
        stats = aggregate_history(model_epoch_rows)
        per_model_stats[model_name] = stats
        overfit_epochs[model_name] = estimate_overfit_epoch(stats)
        curve_path = os.path.join(args.results_dir, "plots", f"{model_name}_curves.png")
        save_model_curves_plot(model_name, stats, curve_path)
        plot_paths[model_name] = curve_path

    combined_plot_path = os.path.join(args.results_dir, "plots", "combined_losses.png")
    save_combined_loss_plot(per_model_stats, combined_plot_path)
    plot_paths["combined"] = combined_plot_path

    report_path = os.path.join(args.results_dir, "benchmark_report.md")
    write_report(report_path, summary_results, per_model_stats, overfit_epochs, plot_paths, args)

    print("\nModel Ranking (mean EER, lower is better):")
    for r in summary_results:
        print(
            f"{r['model']}: mean_eer={r['mean_eer']} "
            f"std={r['std_eer']} best={r['best_eer']} "
            f"(seed {r['best_seed']}, epoch {r['best_epoch']})"
        )

    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title="Model Ranking (mean EER, lower is better)")
        table.add_column("Model")
        table.add_column("Mean EER")
        table.add_column("Std")
        table.add_column("Best EER")
        table.add_column("Best Epoch")
        table.add_column("Best Seed")
        table.add_column("Avg Train (<= best)")
        table.add_column("Avg Dev (<= best)")
        table.add_column("Runs")

        for r in summary_results:
            table.add_row(
                r["model"],
                f"{r['mean_eer']:.4f}" if r["mean_eer"] is not None else "N/A",
                f"{r['std_eer']:.4f}" if r["std_eer"] is not None else "N/A",
                f"{r['best_eer']:.4f}" if r["best_eer"] is not None else "N/A",
                str(r["best_epoch"]),
                str(r["best_seed"]),
                f"{r['avg_train_loss_upto_best']:.4f}" if r.get("avg_train_loss_upto_best") is not None else "N/A",
                f"{r['avg_dev_loss_upto_best']:.4f}" if r.get("avg_dev_loss_upto_best") is not None else "N/A",
                str(r["runs"]),
            )

        Console().print(table)
    except Exception:
        pass

    print(f"\nSaved CSV (runs): {runs_csv_path}")
    print(f"Saved CSV (epochs): {epochs_csv_path}")
    print(f"Saved CSV (ranking): {ranking_csv_path}")
    print(f"Saved report: {report_path}")
    if os.path.exists(plot_path):
        print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
