"""Anomaly detection via CNN2D embeddings + One-Class SVM / GMM.

Uses the trained supervised CNN2D as a feature extractor:
1. Extract embeddings from bonafide-only training data.
2. Fit a One-Class SVM and/or GMM on those embeddings.
3. At test time, score new samples by their distance from the
   learned "normal" distribution and compute EER.
"""

import argparse
from typing import List, Tuple

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from dataloaders import make_loader
from evaluation import calculate_eer
from model import CNN2D


# ── helpers ─────────────────────────────────────────────────────────

def resolve_device(device_arg: str | None) -> str:
    if device_arg:
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(checkpoint_path: str, device: str,
               in_features: int = 180, dropout: float = 0.2) -> CNN2D:
    model = CNN2D(in_features=in_features, dropout=dropout)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model


def extract_embeddings(model: CNN2D, dataloader, device: str,
                       swap_tf: bool = True,
                       bonafide_only: bool = False,
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Extract embeddings and labels from a labeled dataloader."""
    embeddings: List[np.ndarray] = []
    labels: List[float] = []
    with torch.no_grad():
        for features, batch_labels in dataloader:
            features = features.to(device)
            if swap_tf:
                features = features.transpose(1, 2)
            _, emb = model(features, return_embedding=True)
            emb_np = emb.cpu().numpy()
            lbl_np = batch_labels.numpy()

            if bonafide_only:
                mask = lbl_np == 1.0
                emb_np = emb_np[mask]
                lbl_np = lbl_np[mask]

            embeddings.append(emb_np)
            labels.append(lbl_np)

    return np.concatenate(embeddings), np.concatenate(labels)


# ── main ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Embedding-based anomaly detection using CNN2D features."
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to trained CNN2D checkpoint")
    p.add_argument("--train-features", default="data/train/features.pkl")
    p.add_argument("--train-labels", default="data/train/labels.pkl")
    p.add_argument("--dev-features", default="data/dev/features.pkl")
    p.add_argument("--dev-labels", default="data/dev/labels.pkl")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None)
    p.add_argument("--in-features", type=int, default=180)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--swap-tf", action="store_true", default=True)
    p.add_argument("--no-swap-tf", dest="swap_tf", action="store_false")
    p.add_argument("--gmm-components", type=int, default=8,
                   help="Number of GMM components (default: 8)")
    p.add_argument("--svm-nu", type=float, default=0.05,
                   help="OneClassSVM nu parameter (default: 0.05)")
    p.add_argument("--svm-kernel", default="rbf",
                   help="OneClassSVM kernel (default: rbf)")
    return p.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)

    model = load_model(args.checkpoint, device,
                       in_features=args.in_features,
                       dropout=args.dropout)

    # ── Extract bonafide-only training embeddings ───────────────────
    print("Extracting bonafide-only training embeddings...")
    train_loader = make_loader(args.train_features, args.train_labels,
                               batch_size=args.batch_size, shuffle=False)
    train_emb, _ = extract_embeddings(model, train_loader, device,
                                      swap_tf=args.swap_tf,
                                      bonafide_only=True)
    print(f"  Bonafide training embeddings: {train_emb.shape}")

    # ── Extract dev embeddings (both classes) ───────────────────────
    print("Extracting dev embeddings (both classes)...")
    dev_loader = make_loader(args.dev_features, args.dev_labels,
                             batch_size=args.batch_size, shuffle=False)
    dev_emb, dev_labels = extract_embeddings(model, dev_loader, device,
                                             swap_tf=args.swap_tf,
                                             bonafide_only=False)
    print(f"  Dev embeddings: {dev_emb.shape}")

    # ── Normalize embeddings ────────────────────────────────────────
    scaler = StandardScaler()
    train_emb_scaled = scaler.fit_transform(train_emb)
    dev_emb_scaled = scaler.transform(dev_emb)

    # ── One-Class SVM ───────────────────────────────────────────────
    print(f"\nFitting One-Class SVM (nu={args.svm_nu}, kernel={args.svm_kernel})...")
    ocsvm = OneClassSVM(kernel=args.svm_kernel, nu=args.svm_nu)
    ocsvm.fit(train_emb_scaled)

    # decision_function: positive = inlier, negative = outlier
    svm_scores = ocsvm.decision_function(dev_emb_scaled)
    svm_eer, svm_thr = calculate_eer(svm_scores.tolist(), dev_labels.tolist())
    print(f"  One-Class SVM  EER = {svm_eer:.6f}  threshold = {svm_thr:.6f}")

    # ── GMM (with PCA to handle high-dim embeddings) ──────────────
    from sklearn.decomposition import PCA

    n_pca = min(256, train_emb_scaled.shape[1], train_emb_scaled.shape[0])
    print(f"\nReducing to {n_pca} dims via PCA for GMM...")
    pca = PCA(n_components=n_pca, random_state=42)
    train_pca = pca.fit_transform(train_emb_scaled)
    dev_pca = pca.transform(dev_emb_scaled)

    print(f"Fitting GMM (n_components={args.gmm_components})...")
    gmm = GaussianMixture(n_components=args.gmm_components,
                          covariance_type="full",
                          reg_covar=1e-4,
                          random_state=42)
    gmm.fit(train_pca)

    # score_samples: log-likelihood; higher = more normal
    gmm_scores = gmm.score_samples(dev_pca)
    gmm_eer, gmm_thr = calculate_eer(gmm_scores.tolist(), dev_labels.tolist())
    print(f"  GMM            EER = {gmm_eer:.6f}  threshold = {gmm_thr:.6f}")

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Embedding Anomaly Detection Results (dev set)")
    print(f"  One-Class SVM  EER = {svm_eer:.6f}")
    print(f"  GMM            EER = {gmm_eer:.6f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
