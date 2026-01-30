import numpy as np
import torch


def calculate_eer(scores, labels):
    """
    Compute Equal Error Rate (EER) from scores and labels.
    Borrowed from scripts/evaluation.py so you can do quick model-level checks.
    """
    scores_np = np.array(scores)
    labels_np = np.array(labels)

    sorted_indices = np.argsort(scores_np)
    sorted_scores = scores_np[sorted_indices]
    sorted_labels = labels_np[sorted_indices]

    n_bonafide = np.sum(labels_np)
    n_spoof = len(labels_np) - n_bonafide

    if n_bonafide == 0 or n_spoof == 0:
        return 0.0, 0.0

    false_accept_rate = np.concatenate(
        [[1.0], (n_spoof - np.cumsum(sorted_labels == 0)) / n_spoof]
    )
    false_reject_rate = np.concatenate(
        [[0.0], np.cumsum(sorted_labels == 1) / n_bonafide]
    )

    eer_idx = np.argmin(np.abs(false_accept_rate - false_reject_rate))
    eer = (false_accept_rate[eer_idx] + false_reject_rate[eer_idx]) / 2.0

    threshold_epsilon = 1e-6
    if eer_idx == 0:
        threshold = sorted_scores[0] - threshold_epsilon
    elif eer_idx == len(sorted_scores):
        threshold = sorted_scores[-1] + threshold_epsilon
    else:
        threshold = sorted_scores[eer_idx - 1]

    return float(eer), float(threshold)


def evaluate(model, dataloader, criterion=None, device="cpu"):
    """
    Run model on a labeled dataloader and return metrics and raw outputs.

    Returns:
        metrics: dict with avg_loss (if criterion provided), eer, threshold
        scores: list of model scores (logits)
        labels: list of ground-truth labels
    """
    model.eval()
    scores = []
    labels = []
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for features, batch_labels in dataloader:
            # TODO: move to device
            # features = features.to(device)
            # batch_labels = batch_labels.to(device)

            # TODO: forward pass (logits shape: [B, 1] or [B])
            # logits = model(features)

            # TODO: align shapes for loss
            # logits = logits.squeeze(-1)

            # TODO: compute loss if criterion is provided
            # if criterion is not None:
            #     loss = criterion(logits, batch_labels)
            #     total_loss += loss.item() * batch_labels.size(0)
            #     total_count += batch_labels.size(0)

            # TODO: collect scores/labels for EER
            # scores.extend(logits.detach().cpu().tolist())
            # labels.extend(batch_labels.detach().cpu().tolist())
            pass

    avg_loss = (total_loss / total_count) if total_count > 0 else None
    eer, threshold = (None, None)
    if scores and labels:
        eer, threshold = calculate_eer(scores, labels)

    metrics = {
        "avg_loss": avg_loss,
        "eer": eer,
        "threshold": threshold,
    }
    return metrics, scores, labels


if __name__ == "__main__":
    # TODO: add a quick check here if you want to run evaluation directly.
    # Example:
    # - load a saved model checkpoint
    # - build a dev dataloader
    # - call evaluate(...)
    pass
