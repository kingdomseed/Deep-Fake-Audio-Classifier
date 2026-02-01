import argparse
import subprocess

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from model import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Generate prediction.pkl from a model checkpoint.")
    parser.add_argument("--features", required=True, help="Path to features.pkl")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--model", required=True, choices=["cnn1d", "cnn2d", "cnn2d_spatial", "crnn", "crnn2"])
    parser.add_argument("--out", required=True, help="Output path for prediction.pkl")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None, help="cuda, mps, or cpu")
    parser.add_argument("--in-features", type=int, default=321)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--pool-bins", type=int, default=1)
    parser.add_argument("--apply-sigmoid", action="store_true", default=True)
    parser.add_argument("--no-apply-sigmoid", action="store_true", default=False)
    parser.add_argument("--swap-tf", action="store_true", help="swap time and feature dimensions (T <-> F)")
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> str:
    if device_arg:
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_model_kwargs(args):
    if args.model == "cnn1d":
        return {"in_channels": args.in_features, "dropout": args.dropout, "pool_bins": args.pool_bins}
    if args.model in {"cnn2d", "cnn2d_spatial"}:
        return {"in_features": args.in_features, "dropout": args.dropout}
    if args.model in {"crnn", "crnn2"}:
        return {"in_features": args.in_features, "dropout": args.dropout}
    return {}


class FeatureOnlyDataset(Dataset):
    def __init__(self, features_df):
        self.features = features_df["features"].reset_index(drop=True)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features.iloc[idx].float()


def main():
    args = parse_args()
    device = resolve_device(args.device)

    apply_sigmoid = False if args.no_apply_sigmoid else args.apply_sigmoid

    model_kwargs = build_model_kwargs(args)
    model = build_model(args.model, **model_kwargs).to(device)

    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    features_df = pd.read_pickle(args.features)
    if "uttid" not in features_df.columns:
        raise ValueError("features.pkl must contain 'uttid'")

    dataset = FeatureOnlyDataset(features_df)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    predictions = []
    with torch.no_grad():
        for features in dataloader:
            features = features.to(device)
            if args.swap_tf:
                features = features.transpose(1, 2)
            logits = model(features).squeeze(-1)
            if apply_sigmoid:
                scores = torch.sigmoid(logits)
            else:
                scores = logits
            predictions.extend(scores.detach().cpu().tolist())

    if len(predictions) != len(features_df):
        raise ValueError("Number of predictions does not match number of rows in features.pkl")

    pred_df = pd.DataFrame(
        {
            "uttid": features_df["uttid"].values,
            "predictions": predictions,
        }
    )
    pred_df.to_pickle(args.out)


if __name__ == "__main__":
    main()
