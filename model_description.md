# Model Description - Jason Holt

# Data

This project uses several data augmentation techniques from `src/augmentation.py` to make the model more robust:

- **SpecAugment (Time & Feature Masks):** I black out random chunks of time (20%) and frequency ranges (10%). This forces the model to learn the overall "vibe" of the voice rather than just looking for specific small glitches (local artifacts).
- **Circular Time Shift:** I shift the audio starting point so the model doesn't over-memorize patterns that only happen at the very beginning or end.
- **Channel Drop:** I randomly zero out some feature channels to help the model stay accurate even if some data is missing or messy.
- **Gaussian Jitter:** I add a tiny bit of random noise to the audio so the model doesn't get "spoiled" by training data that is too clean.

# Model

The model in `src/model.py` is a 2D CNN designed to find patterns in the time-feature grid of the audio. The input is a single-channel 2D grid of LFCC, delta, and delta-delta coefficients (180 features x 321 time steps). A 3x3 convolutional filter slides across both dimensions looking for local patterns, then a 2x1 average pooling layer compresses the time axis while keeping all feature dimensions intact.

- **Hierarchy:** It uses three "convolutional blocks" that get more complex as they go (starting with 32 focus points, then 64, then 128). This helps the model learn simple sounds first and then more complex speech patterns later.
- **Blocks:** Each block uses Convolution, Batch Normalization (for stability), ReLU (to learn patterns), and Average Pooling.
- **Time Focus:** I apply pooling specifically to the time dimension. This shrinks the timeline but keeps the frequency details sharp, which is important for spotting deepfakes.

# Loss

I used **Binary Cross Entropy (BCE)** loss with **Label Smoothing (0.05)**.

- Instead of telling the model that every sample is "100% Real" or "100% Fake," I "soften" the targets. This prevents the model from becoming over-confident, and possibly better at handling new data.

# Others

- **Optimization:** I used the AdamW optimizer with a learning rate scheduler that lowers the learning rate if the model stops improving.
- **Early Stopping:** Training was stopped after 8 epochs of no improvement to ensure the model didn't start "memorizing” training data.
- **Verification:** This final model achieved **0.00 EER on Test 1** while maintaining a safe, spread-out score distribution. Whether this stands up to the more challenging data remains to be seen.
