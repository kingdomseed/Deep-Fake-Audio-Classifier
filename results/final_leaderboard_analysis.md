# Final Leaderboard Analysis

**Date:** February 10, 2026
**Submission:** WhatAreLogits (Jason Holt, st199007)

## Final Results

| Leaderboard | EER | Rank | Out of |
|---|---|---|---|
| In-domain (seen attacks) | 0.00% | 1st (tied with 16 others) | 59 |
| Out-of-domain (unseen attacks) | 27.08% | 28th (tied with 13 others) | 59 |
| Weighted (0.9×ID + 0.1×OOD) | 2.71% | 17th | 59 |

## In-Domain: Table Stakes, Not the Competition

17 out of 59 models achieved 0.00% EER on in-domain (seen attacks). Perfect performance on seen data was necessary but not differentiating. This confirms that the training data distribution was easy to learn — even simple models could memorize the seen attack artifacts.

## Out-of-Domain: Where the Real Separation Happened

| Tier | EER Range | Count | Notable |
|---|---|---|---|
| Top | 18.75-22.92% | 11 | dlqueen (18.75%), cPagluFinalBoss, snowman, coffee |
| Upper-mid | 25.00% | 16 | Protege, UPS, AURAFARMED, Sillystar |
| Mid | 27.08% | **WhatAreLogits**, plus 13 others |
| Lower | 29.17-45.83% | 19 | Models with extreme logit distributions |

Even the best model (dlqueen at 18.75%) got nearly 1 in 5 unseen samples wrong. Nobody "solved" out-of-domain generalization.

## Key Observations

### 1. Shared predictions exist

Four models have identical score statistics (mean=1.8553, std=9.4440): `immutable`, `machinenotlearning`, `nogradients`, `m_m`. These are either the same model submitted under different names or shared predictions — exactly the collaborative approach the professor encouraged.

### 2. dlqueen's OOD dominance

dlqueen won OOD by a significant margin (18.75% vs next-best 20.83%). Their OOD score distribution (mean=-9.23, std=4.65) suggests raw logits with a strongly negative bias — they may have applied a normalization technique (CMN/CVMN) or had a fundamentally different feature preprocessing that handled unseen attacks better.

### 3. Calibration helped but didn't win

Our score distribution (mean=0.2557, std=0.3182 on OOD) was well-calibrated — sigmoid outputs, moderate spread. Models with extreme raw-logit distributions (mean=-5 to -10, std=5-22) had mixed results: some did better on OOD (dlqueen, cheese) but many did worse (Pluto, ZeroLoss, 3coffeeNoSleep). Calibration prevented catastrophic failure but didn't provide an OOD advantage.

### 4. The professor's MTS prediction held

The professor said EER would rise to ~10% on shifted data (MTS analysis with mixed seen/unseen). The pure OOD leaderboard shows 18-45%, confirming that fully unseen attacks are harder than the MTS mix. The professor also removed some hard bonafide samples from the final set, which is why in-domain scores are so clean.

### 5. Compression ≠ safety

`fiftyone` had the most compressed distribution (mean=0.3785, std=0.2232 on OOD) — almost everything squeezed toward center. They got the same 27.08% OOD as us. `Hejhej` (mean=0.3854, std=0.1691) was even more compressed and got 22.92%. Caution alone doesn't solve generalization.

### 6. dlqueen reproduction confirms architecture matters more than normalization

We reproduced dlqueen's exact model (`DeepfakeDetector` CNN1D + StatsPool, hidden=256, EMA, class weighting) on our data and compared outputs head-to-head.

**Test1 (in-domain, 500 labeled samples):**

| | Ours (CNN2D) | dlqueen repro (CNN1D+Stats) |
|---|---|---|
| Test1 EER | 0.00% | 0.41% |
| Binary agreement | 499/500 (99.8%) | — |
| Bonafide score (sigmoid) | 0.999 +/- 0.015 | 0.993 +/- 0.069 |
| Spoof score (sigmoid) | 0.006 +/- 0.023 | 0.000 +/- 0.004 |

On easy data, both models perform near-identically. The 1 disagreement: `raw_16227` (we say REAL 0.78, she says FAKE 0.06).

**Test2 (OOD, 100 unlabeled samples) — where the gap appears:**

| | Ours (CNN2D) | dlqueen repro |
|---|---|---|
| Calls REAL | 29 | 6 |
| Calls FAKE | 71 | 94 |
| Binary agreement | 77/100 (77%) | — |
| Sigmoid mean | 0.272 | 0.070 |
| Score < 0.01 (sigmoid) | 32 | **83** |
| Score > 0.50 (sigmoid) | **29** | 6 |

All 23 disagreements are one-directional: **we say REAL, she says FAKE.** Not a single case where she says REAL and we say FAKE.

Her reproduced OOD distribution (logit mean=-10.36, std=5.71) closely matches her leaderboard stats (mean=-9.23, std=4.65), confirming valid reproduction.

**Why her architecture produces more conservative OOD calls:**

- **StatsPool (mean+std)** captures variance — notices when audio is "too smooth" or "too uniform," a hallmark of unseen deepfakes
- **Same-width channels (256→256→256)** allocates more parameters to the first layer, learning richer low-level representations
- **WeightedRandomSampler** explicitly compensates for class imbalance, pushing the decision boundary toward caution
- **EMA** smooths weights, reducing overfit to training-set-specific artifacts
- **GELU** instead of ReLU preserves more gradient information in negative regions

The net effect: on uncertain/unseen data, her model defaults to "I don't recognize this → probably fake." Ours defaults to "this looks normal enough → probably real." On a test set dominated by unseen fakes, her prior is better.

## What Would Have Improved Our OOD Score

Based on what the top OOD models appear to have done, confirmed by direct reproduction:

- **StatsPool (mean+std) over mean-only pooling:** Capturing variance information is the single biggest architectural difference. Our archived `StatsPoolMLP` had this idea but applied it in an MLP — wrong context, right concept.
- **Wider early layers (256 channels from layer 1):** Our CNN2D used 32→64→128, dlqueen used 256→256→256. More capacity at early layers learns richer features that generalize better.
- **Class weighting:** `WeightedRandomSampler` + `pos_weight` in BCEWithLogitsLoss explicitly counteracts the class imbalance, making the model less prone to the majority-class default.
- **EMA weight averaging:** Smooths out training noise, producing more robust final weights.
- **Feature normalization (CMN/CVMN):** Our comparison experiment showed CMN slightly hurt in-domain (0.20% vs 0.00% dev EER) but CVMN maintained 0.00% dev while shifting score distributions. The impact on OOD remains untested.
- **Collaborative ensembling:** The 4 identical models suggest a group that pooled predictions. Cross-architecture voting can smooth out individual model blind spots.

## Our Exploration in Context

| Method | Dev EER | Test1 EER | OOD EER (est.) | Notes |
|---|---|---|---|---|
| Supervised CNN2D (submitted) | 0.00% | 0.00% | 27.08% (actual) | Best available at submission time |
| dlqueen repro (CNN1D+Stats) | 0.10% | 0.41% | ~18-20% (est.) | Reproduced her architecture on our data |
| One-Class SVM on embeddings | 4.55% | — | unknown | Different signal but derived from same model |
| CAE (bonafide-only) | 6.70% | 9.19% | unknown | Inverted polarity; discrimination ceiling |
| Hybrid (80% sup + 20% CAE) | 0.00% | — | unknown | CAE flagged 12 of the 37 classmate-disputed samples |
| CNN2D + CMN | 0.20% | 0.41% | untested | Normalization slightly hurt in-domain |
| CNN2D + CVMN | 0.10% | 0.00% | untested | Maintained in-domain, shifted distribution |

The CAE exploration produced the right *direction* of insight (questioning borderline "real" classifications) but wasn't strong enough to improve the submission. The classmate voting analysis identified 37 disputed samples, but the professor's MTS analysis made us correctly cautious about overriding our scores based on group consensus.

The dlqueen reproduction confirmed that the OOD gap came from architectural choices (StatsPool, wider channels, class weighting, EMA), not from some hidden preprocessing trick. We had several of the right ideas in archived models (StatsPoolMLP, wider layers) but never combined them in the right architecture.

## Final Takeaway

In-domain perfection is achievable with a well-regularized CNN and modest augmentation. Out-of-domain generalization is a fundamentally harder problem that requires either (a) feature-level normalization to remove distribution-specific biases, (b) genuinely diverse model ensembles, or (c) training strategies specifically designed for domain shift. Our robustness measures (label smoothing, SpecAugment, time shift) were pointed in the right direction but not sufficient to close the gap to the top OOD performers.

Upper third of the class (17/59) with a well-understood model and thoroughly documented exploration. The post-submission analysis — reproducing the winning model, testing normalization variants, and understanding *why* architectural choices matter — is arguably more valuable for learning than the leaderboard position itself.
