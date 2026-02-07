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
- **v6 (200 epochs):** Dev EER = 7.00%, Test1 EER = 9.60%
- **v7 (300 epochs, gentler LR schedule):** Still training at time of commit

### Tier 3: Hybrid Ensemble (supervised + CAE)
- Alpha sweep: `score = alpha * supervised + (1-alpha) * cae_anomaly`
- On dev: any alpha >= 0.55 gives 0.00% EER (supervised dominates)
- On final test (no labels): 98.8% class agreement at alpha=0.80, 72.7% at alpha=0.00
- **Takeaway:** The CAE produces a genuinely different signal on ~27% of the harder test data, but we cannot verify whether it's *better* without labels.

## Key Findings

### 1. Deepfakes are simpler than real speech
The CAE reconstructs fakes *better* than reals (MSE 0.24 vs 0.47, ratio 0.52x). This inverts the standard anomaly detection assumption. Real human speech has more prosodic complexity, pitch variation, and natural irregularity. Deepfakes are smoother and more regular — closer to "average audio."

### 2. More training doesn't help the CAE discriminate
| Run | Val MSE | Dev EER | Test1 EER |
|---|---|---|---|
| v5 (80ep) | 0.469 | **6.70%** | **9.19%** |
| v6 (200ep) | 0.403 | 7.00% | 9.60% |

Better reconstruction (lower MSE) doesn't improve discrimination. The model gets better at reconstructing *everything* equally. The spoof/bonafide MSE ratio stayed flat at ~0.52x.

### 3. The LR scheduler matters more than epoch count
v6's scheduler (patience=3) decayed the LR to 1e-6 too quickly, freezing the model. v7 used patience=7, keeping the LR at 2.5e-5 through epoch 170+. The reconstruction loss was lower in v7 but EER results pending.

### 4. The professor's "hacking" is semi-supervised, not unsupervised
From the Jan 27 lecture: the professor described using unsupervised methods to **infer pseudo-labels** on the test set, then using cross-model voting to refine them. This is different from building a standalone anomaly detector. The pipeline is:
1. Multiple different models produce scores on unlabeled data
2. Vote/average across models to estimate labels
3. Use pseudo-labels for further training or calibration

## What We Learned (Concepts)

- **Why bottlenecks matter:** A single linear layer compressing 112K -> 512 creates an information cliff where gradients can't propagate. Distributed spatial compression through conv layers works.
- **Feature normalization is critical:** LFCC features span [-50, +10]. Without z-score normalization, MSE is dominated by high-magnitude features.
- **Anomaly detection assumptions can invert:** When the "anomaly" (deepfakes) is simpler than normal (real speech), reconstruction error goes the wrong way.
- **Reconstruction quality != discrimination quality:** More epochs improve MSE but don't improve the gap between classes.

## Open Questions

1. Would scoring on **delta/delta-delta bands only** (features 60-180) be more discriminative? Those capture temporal *changes* where real speech should differ most from smooth fakes.
2. Would a **VAE** with KL regularization force the latent space to be more structured, preventing the model from learning a "reconstruct anything" representation?
3. The professor's semi-supervised voting approach: with our supervised model's scores + CAE's anomaly scores + classmate's k-means clusters, could we triangulate pseudo-labels on the final test set?
4. Is the CAE's inverted polarity actually more robust on truly unseen attacks? We can't answer this without harder test labels.

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
