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

## What Would Have Improved Our OOD Score

Based on what the top OOD models appear to have done:
- **Feature normalization (CMN/CVMN):** Several top OOD models likely applied cepstral mean/variance normalization, which removes channel-specific biases that differ between seen and unseen recording conditions.
- **Collaborative ensembling:** The 4 identical models suggest a group that pooled predictions. Cross-architecture voting can smooth out individual model blind spots.
- **Different architectural choices:** Some top models may have used architectures less prone to memorizing seen-attack-specific artifacts.

## Our Exploration in Context

| Method | Dev EER | Test1 EER | OOD EER (est.) | Notes |
|---|---|---|---|---|
| Supervised CNN2D (submitted) | 0.00% | 0.00% | 27.08% (actual) | Best available at submission time |
| One-Class SVM on embeddings | 4.55% | — | unknown | Different signal but derived from same model |
| CAE (bonafide-only) | 6.70% | 9.19% | unknown | Inverted polarity; discrimination ceiling |
| Hybrid (80% sup + 20% CAE) | 0.00% | — | unknown | CAE flagged 12 of the 37 classmate-disputed samples |

The CAE exploration produced the right *direction* of insight (questioning borderline "real" classifications) but wasn't strong enough to improve the submission. The classmate voting analysis identified 37 disputed samples, but the professor's MTS analysis made us correctly cautious about overriding our scores based on group consensus.

## Final Takeaway

In-domain perfection is achievable with a well-regularized CNN and modest augmentation. Out-of-domain generalization is a fundamentally harder problem that requires either (a) feature-level normalization to remove distribution-specific biases, (b) genuinely diverse model ensembles, or (c) training strategies specifically designed for domain shift. Our robustness measures (label smoothing, SpecAugment, time shift) were pointed in the right direction but not sufficient to close the gap to the top OOD performers.

Upper third of the class (17/59) with a well-understood model and thoroughly documented exploration. Not a bad outcome.
