# Model Description

# Data

src/augmentation.py
Includes: 
- time and feature masks - blacked out chunks of time (20%) and blacked out 10% random ranges of frequencies.
- time shift - shifted audio cyclically so patterns that only appear at the beginning or end are not over emphasized in those positions.
- channel_drop - Randomly zero out specific feature Chanels to make the model more robust against missing data.
- gaussian jitter -  Added a tiny bit of random white noise in case the audio was too clean.

# Model

src/model.py
2D CNN
- A stacked architecture consisting of three convolutional blocks using hierarchy starting from simple to complex (32 channels, 64 channels, and then 128 channels).
- A stack: Conv2d, BatchNom2d, ReLU, AvgPool2d, Dropout
	- Average pooling is applied to the temporal dimension.
- Final stack: Conv2d, BatchNorm2d, ReLU
- A final Linear Layer for classification

# Loss

Binary Cross Entropy (BCE) loss with Label Smoothing at 0.05 to try and help the model not be overconfident by telling it that not every sample is 100% real or 100% fake.

# Others

Optimization: AdamW with a learning rate scheduler, early stopping after 8 epochs (50 was the target), and checked the score distribution (thanks to Pengcheng) to ensure we had some variation in the learning. 