# Anomaly Detection Exploration — Findings

**Date:** February 7, 2026
**Context:** Post-submission exploration of unsupervised methods for audio deepfake detection, building on a supervised CNN2D that achieved 0.00% EER on dev and test1.

## What We Tried

### Tier 1A: Ensemble Averaging (4 existing checkpoints)

- **Result:** 0.00% EER on dev — no improvement over best single model
- **Takeaway:** All models make the same mistakes on this data. Ensembling only helps when models are *independently wrong*.

### Tier 1B: Embedding Extraction + One-Class SVM / GMM

- Extracted 23,040-dim embeddings from the supervised CNN2D's penultimate layer
- Fit One-Class SVM on bonafide-only embeddings
- **Result:** OC-SVM = 4.55% EER on dev, GMM = 25.1% EER
- **Takeaway:** The supervised model's internal representation already separates real/fake cleanly (your classmate's k-means PCA plot confirms this). The OC-SVM works because the embedding space has structure, but it's borrowing from the supervised model's learned features — not truly unsupervised.

### Tier 2: Convolutional Autoencoder (bonafide-only training)

- **Previous attempt failed** (flatlined at epoch 2) due to a 220:1 linear bottleneck — recovered architecture from `.pyc` bytecode
- **Fix:** Fully convolutional encoder/decoder, spatial bottleneck (256, 20, 11), ConvTranspose2d for learned upsampling, per-feature z-score normalization
- **v5 (80 epochs):** Dev EER = 6.70%, Test1 EER = 9.19%
- **v6 (200 epochs, scheduler patience=3):** Dev EER = 7.00%, Test1 EER = 9.60%
- **v7 (300 epochs, scheduler patience=7):** Dev EER = 7.00%, Test1 EER = 9.60%

### Tier 3: Hybrid Ensemble (supervised + CAE)

- Alpha sweep: `score = alpha * supervised + (1-alpha) * cae_anomaly`
- On dev: any alpha >= 0.55 gives 0.00% EER (supervised dominates)
- On final test (no labels): 98.8% class agreement at alpha=0.80, 72.7% at alpha=0.00
- **Takeaway:** The CAE produces a genuinely different signal on ~27% of the harder test data, but we cannot verify whether it's *better* without labels.

## Key Findings

### 1. Deepfakes are simpler than real speech

The CAE reconstructs fakes *better* than reals (MSE 0.24 vs 0.47, ratio 0.52x). This inverts the standard anomaly detection assumption. Real human speech has more prosodic complexity, pitch variation, and natural irregularity. Deepfakes are smoother and more regular — closer to "average audio."

### 2. More training doesn't help the CAE discriminate

| Run | Val MSE | Dev EER | Test1 EER | Spoof/Bonafide Ratio |
|---|---|---|---|---|
| v5 (80ep, sched patience=3) | 0.469 | **6.70%** | **9.19%** | 0.52x |
| v6 (200ep, sched patience=3) | 0.403 | 7.00% | 9.60% | 0.52x |
| v7 (300ep, sched patience=7) | 0.373 | 7.00% | 9.60% | 0.52x |

Better reconstruction (lower MSE) doesn't improve discrimination. The model gets better at reconstructing *everything* equally. The spoof/bonafide MSE ratio is stuck at ~0.52x across all runs. v5 at 80 epochs was actually the best discriminator despite having the worst reconstruction quality.

### 3. The LR scheduler matters more than epoch count

v6's scheduler (patience=3) decayed the LR to 1e-6 by epoch ~160, freezing the model early. v7 used patience=7, which kept the LR at 2.5e-5 through epoch 170+ and only hit 6.25e-6 at epoch 292. Despite reaching a significantly lower val MSE (0.373 vs 0.403), the EER was identical. The gentler schedule produced better reconstruction but no better discrimination — confirming that the CAE's limitation is structural, not a training issue.

### 4. On tiny datasets, simpler models win

Archived models (MeanPoolMLP, StatsPoolMLP, CRNN, CRNN2, CNN2D_Robust with SE attention) all performed worse than the simple 3-block CNN2D. The MLPs destroyed temporal locality by mean-pooling over time. The CRNNs and attention models overfitted — more parameters + 6,400 training samples = memorization. The CNN1D wasn't submitted because its score distribution was too extreme (overconfident), making it fragile under distribution shift. Label smoothing + calibrated scores > raw accuracy.

### 5. Semi-supervised "hacking" requires independent signals

Using the OC-SVM or CAE as pseudo-label generators for fine-tuning is largely circular when they're derived from the same base model's embeddings. The professor's voting approach requires genuinely different architectures from different people. A single student's models share too much underlying representation for the votes to be independent.

### 6. The professor's "hacking" is semi-supervised, not unsupervised

From the Jan 27 lecture: the professor described using unsupervised methods to **infer pseudo-labels** on the test set, then using cross-model voting to refine them. This is different from building a standalone anomaly detector. The pipeline is:

1. Multiple different models produce scores on unlabeled data
2. Vote/average across models to estimate labels
3. Use pseudo-labels for further training or calibration

## What We Learned (Concepts)

- **Why bottlenecks matter:** A single linear layer compressing 112K -> 512 creates an information cliff where gradients can't propagate. Distributed spatial compression through conv layers works.
- **Feature normalization is critical:** LFCC features span [-50, +10]. Without z-score normalization, MSE is dominated by high-magnitude features.
- **The data is LFCC, not spectrograms:** Features are [60 LFCC + 60 delta + 60 delta-delta] cepstral coefficients, not raw spectrograms. The 2D CNN "image analogy" works structurally but the value semantics are different.
- **Anomaly detection assumptions can invert:** When the "anomaly" (deepfakes) is simpler than normal (real speech), reconstruction error goes the wrong way.
- **Reconstruction quality != discrimination quality:** More epochs improve MSE but don't improve the gap between classes.
- **Score calibration matters more than raw accuracy:** The CNN1D had similar or better accuracy but was rejected for submission because its score distribution was too extreme. Overconfident models are fragile under distribution shift. Label smoothing (0.05) was key to the submitted model's robustness.
- **The OC-SVM is already a pseudo-label generator:** At 4.55% EER, `ocsvm.predict()` returns +1/-1 labels that are ~95.5% accurate. But fine-tuning on those pseudo-labels is circular when they come from the same base model's embeddings.
- **Semi-supervised requires collaboration:** The professor explicitly blessed unsupervised "hacking" but emphasized that the voting approach requires genuinely different models from different people. A single student's models share too much representation for independent votes.

### 7. Normalization helps less than architecture on this data

| Mode | Dev EER | Test1 EER | Score Mean | Score Std |
|---|---|---|---|---|
| Raw (submitted) | 0.00% | 0.00% | 0.4072 | 0.4568 |
| CMN | 0.20% | 0.41% | 0.2755 | 0.3692 |
| CVMN | 0.10% | 0.00% | 0.4489 | 0.4605 |

CMN slightly hurt performance. CVMN maintained near-perfect in-domain accuracy while shifting the score distribution. Neither is a game-changer on the same CNN2D architecture. The bigger lever for OOD is architectural (StatsPool, wider channels, EMA) as confirmed by reproducing dlqueen's model.

### 8. Right ideas, wrong combinations

Our archived `StatsPoolMLP` used mean+std pooling (the key ingredient in dlqueen's model) but applied it in an MLP that destroyed temporal locality. Our CNN2D had the right spatial structure but used mean-only pooling. The winning recipe was the combination we never tried: **1D conv + StatsPool + wide channels + EMA + class weighting.**

## Open Questions

1. Would scoring on **delta/delta-delta bands only** (features 60-180) be more discriminative? Those capture temporal *changes* where real speech should differ most from smooth fakes.
2. Would a **VAE** with KL regularization force the latent space to be more structured, preventing the model from learning a "reconstruct anything" representation?
3. The professor's semi-supervised voting approach: with our supervised model's scores + CAE's anomaly scores + classmate's k-means clusters, could we triangulate pseudo-labels on the final test set?
4. Is the CAE's inverted polarity actually more robust on truly unseen attacks? We can't answer this without harder test labels.
5. Would our CNN2D with StatsPool (replace `mean(dim=2)` with mean+std concat) close the gap to dlqueen without changing anything else?

## Archived Models (from `src/archive/models.py`)

| Model | Why It Failed |
|---|---|
| MeanPoolMLP | Destroyed temporal locality — mean over time buries localized artifacts |
| StatsPoolMLP | Same problem but with mean+std+max; still no local structure |
| CNN1DSpatial | `in_channels=321` (time dim) — convolved over wrong axis |
| CRNN (1-layer GRU) | Overfitted — RNN parameters + tiny dataset = memorization |
| CRNN2 (2-layer GRU) | Same overfitting, worse with more RNN layers |
| CNN2D_Robust (SE attention, double conv) | Too many parameters (~5-10x CNN2D); overfitted on 6,400 samples |
| CNN1D (final, not archived) | Worked well but rejected for submission — score distribution too extreme/overconfident |

## Conclusion

The supervised CNN2D with label smoothing and augmentation remains the best single model for this task. The unsupervised exploration (CAE, OC-SVM, ensemble) provided valuable learning about anomaly detection, feature spaces, and model calibration, but did not produce a model worth substituting for the submission.

The fundamental constraint is the tiny dataset (6,400 train, 2,000 dev). On this scale, simple architectures with good regularization beat complex ones, and a single well-tuned supervised model leaves little room for unsupervised methods to improve upon — especially when the "anomaly" (deepfakes) is structurally simpler than the normal class (real speech).

The path to actual improvement would require either (a) genuinely different model architectures from collaborators for voting-based semi-supervised learning, or (b) a fundamentally different loss function (e.g., contrastive, perceptual) that captures higher-level "naturalness" rather than pixel-level reconstruction.

## Post-Exploration: Cross-Model Voting Analysis (Feb 7)

### Classmate ensemble data

Received 6 prediction files from a classmate — all 2DCNN variants with different preprocessing on the final test set:

| File | Method | Est Real | Est Fake |
|---|---|---|---|
| prediction_0 | Baseline (no special method) | 387 | 613 |
| prediction_1 | CVMN (cepstral variance normalization) | 372 | 628 |
| prediction_2 | CMN (cepstral mean normalization) | 400 | 600 |
| prediction_3 | Std dev deleted in forward pass | 384 | 616 |
| prediction_4 | Std dev dropped out by chance | 377 | 623 |
| prediction_5 | (unlabeled 6th variant) | 375 | 625 |

### Voting results (7 models: ours + 6 classmate)

- **92.6% unanimous** (926/1000): 359 all-say-real + 567 all-say-fake
- **74 split-vote samples**: only these are contested
- Our model agrees with majority on **96.0%** of samples
- **37 samples**: we say real, classmate majority says fake
- **3 samples**: we say fake, classmate majority says real
- Our model is the **most generous with "real" labels** (418 vs group average ~383)
- Classmate models agree with each other at 97-99.5%; with us at 94-96%

### CAE caught the same signal (partially)

The hybrid ensemble (alpha=0.80, 20% CAE) flipped 12 samples from real to fake. **All 12 were a perfect subset of the 37 classmate-disputed samples.** Zero false positives — the CAE independently flagged the same suspicious samples, but only the most borderline ones (our scores 0.50-0.55). It missed the 25 where our model was more confident (0.56-0.95).

### Professor's MTS analysis changes the interpretation

From the Feb 6 announcement, the professor revealed the MTS (final test set) results:
- **Mean EER: 10.37%, Median: 9.33%, range: 6-18%** across 21 valid submissions
- Half the deepfakes are unseen attacks, half are seen attacks
- **All models handle new deepfakes well** — confident low scores
- **Most models fail on new bonafide audio** — classifying calm real speech as fake
- Prosody richness ranking: **old bonafide (vivid) > new bonafide (calm/flat) > deepfakes (smoothest)**
- Models with EER >10% show overfitting: overly confident with large score gaps

### Why we did NOT modify the submission

The 37 disputed samples (we say real, classmates say fake) might include new calm bonafide that all models systematically misclassify. The professor explicitly stated the dominant error mode is **false rejection of calm bonafide**, not false acceptance of deepfakes.

The classmate voting consensus is biased by the same systematic error — 6 models trained on vivid bonafide all see calm bonafide as suspiciously smooth. Our model's more generous "real" calls may be partially correct on exactly these samples, thanks to:
- Label smoothing (0.05) preventing overconfidence
- Time masking and time shift augmentation reducing prosody dependence
- Calibrated score distribution (not extreme/overconfident)

Overriding our scores with group consensus could make things **worse** by amplifying the shared bias rather than correcting it. Decision: **keep original submission unchanged.**

## Files Created

| File | Purpose |
|---|---|
| `src/ensemble.py` | Multi-checkpoint ensemble averaging |
| `src/embedding_anomaly.py` | OC-SVM/GMM on CNN2D embeddings |
| `src/model_cae.py` | Fully-convolutional autoencoder (561K params) |
| `src/dataset_cae.py` | Bonafide-only dataset + feature normalization |
| `src/train_cae.py` | CAE training with Rich progress display |
| `src/evaluation_cae.py` | MSE-based anomaly scoring + EER |
| `src/hybrid_ensemble.py` | Supervised + CAE alpha sweep |
| `src/predict_hybrid.py` | Final test prediction + distribution comparison |
| `src/compare_normalization.py` | CMN/CVMN comparison on CNN2D |
| `src/compare_kernels.py` | Kernel size + normalization experiments on CNN1D |
| `src/dlqueen_model.py` | Reproduction of top-ranked classmate's model |
| `results/prediction_dlqueen_test1.pkl` | dlqueen repro predictions on test1 |
| `results/prediction_dlqueen_test2.pkl` | dlqueen repro predictions on test2 (OOD) |
| `results/prediction_ours_test2.pkl` | Our CNN2D predictions on test2 (OOD) |
