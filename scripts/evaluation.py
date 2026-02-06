import sys
import pandas as pd
import numpy as np
import pickle


def calculate_eer(scores, labels):
    scores_np = np.array(scores)
    labels_np = np.array(labels)

    sorted_indices = np.argsort(scores_np)
    sorted_scores = scores_np[sorted_indices]
    sorted_labels = labels_np[sorted_indices]

    n_bonafide = np.sum(labels_np)
    n_spoof = len(labels_np) - n_bonafide

    if n_bonafide == 0 or n_spoof == 0:
        return 0.0, 0.0

    false_accept_rate = np.concatenate([[1.0],
                                        (n_spoof - np.cumsum
                                        (sorted_labels == 0)) / n_spoof])
    false_reject_rate = np.concatenate([[0.0],
                                        np.cumsum(sorted_labels == 1)
                                        / n_bonafide])

    eer_idx = np.argmin(np.abs(false_accept_rate - false_reject_rate))
    eer = (false_accept_rate[eer_idx] + false_reject_rate[eer_idx]) / 2.0

    THRESHOLD_EPSILON = 1e-6
    if eer_idx == 0:
        threshold = sorted_scores[0] - THRESHOLD_EPSILON
    elif eer_idx == len(sorted_scores):
        threshold = sorted_scores[-1] + THRESHOLD_EPSILON
    else:
        threshold = sorted_scores[eer_idx - 1]

    return float(eer), float(threshold)


def confusion_at_threshold(scores, labels, threshold):
    scores_np = np.array(scores)
    labels_np = np.array(labels).astype(int)

    pred = (scores_np > threshold).astype(int)

    tp = int(np.sum((pred == 1) & (labels_np == 1)))
    fn = int(np.sum((pred == 0) & (labels_np == 1)))
    fp = int(np.sum((pred == 1) & (labels_np == 0)))
    tn = int(np.sum((pred == 0) & (labels_np == 0)))

    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    return tp, fp, tn, fn, float(far), float(frr)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: python evaluation.py <prediction.pkl> <labels.pkl>")

    prediction_path = sys.argv[1]
    labels_path = sys.argv[2]

    prediction_df = pd.read_pickle(prediction_path)
    labels_df = pd.read_pickle(labels_path)

    if 'uttid' not in prediction_df.columns or 'predictions' not in prediction_df.columns:
        raise ValueError("prediction.pkl must have 'uttid' and 'predictions' columns")

    if 'uttid' not in labels_df.columns or 'label' not in labels_df.columns:
        raise ValueError("labels.pkl must have 'uttid' and 'label' columns")

    merged = pd.merge(prediction_df, labels_df, on='uttid', how='inner')

    if len(merged) != len(prediction_df) or len(merged) != len(labels_df):
        raise ValueError("uttid mismatch between prediction and labels")

    scores = merged['predictions'].values
    labels = merged['label'].values

    eer, threshold = calculate_eer(scores, labels)

    tp, fp, tn, fn, far, frr = confusion_at_threshold(scores, labels, threshold)

    print(f"EER: {eer:.6f}")
    print(f"Threshold: {threshold:.6f}")
    print(f"TP: {tp}  FP: {fp}  TN: {tn}  FN: {fn}")
    print(f"FAR: {far:.6f}  FRR: {frr:.6f}")
