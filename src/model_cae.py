"""Fully-convolutional Autoencoder for anomaly-based deepfake detection.

Trained on bonafide-only spectrograms, the model learns to reconstruct
"normal" audio.  Deepfakes produce higher reconstruction error (MSE).

Architecture fixes vs. the previous attempt (recovered from bytecode):
 1. No linear bottleneck — compression is purely spatial via 4 conv blocks.
 2. ConvTranspose2d for learned upsampling instead of bilinear + Conv2d.
 3. Expects normalised input (zero mean, unit variance per feature dim).

Input:  (B, T, F) with T=321, F=180  (after swap_tf)
Internal: (B, 1, T, F) -> encoder -> (B, 256, 20, 11) -> decoder -> (B, 1, 321, 180)
Output: (reconstruction, latent_map)
"""

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """Fully-convolutional autoencoder with spatial bottleneck."""

    def __init__(self, base_channels: int = 32):
        super().__init__()

        bc = base_channels  # 32

        # ── Encoder ─────────────────────────────────────────────────
        # Each block halves both spatial dims via AvgPool2d(2).
        # (B,1,321,180) -> pad to even -> pool chain
        # After 4 pools: T: 321->160->80->40->20  F: 180->90->45->22->11
        self.encoder = nn.Sequential(
            # Block 1:  1 -> 32
            nn.Conv2d(1, bc, kernel_size=3, padding=1),
            nn.BatchNorm2d(bc),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),          # (B,32,160,90)

            # Block 2: 32 -> 64
            nn.Conv2d(bc, bc * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(bc * 2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),          # (B,64,80,45)

            # Block 3: 64 -> 128
            nn.Conv2d(bc * 2, bc * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(bc * 4),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),          # (B,128,40,22)

            # Block 4: 128 -> 256   (bottleneck)
            nn.Conv2d(bc * 4, bc * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(bc * 8),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),          # (B,256,20,11)
        )

        # ── Decoder ─────────────────────────────────────────────────
        # Mirror the encoder using ConvTranspose2d for learned upsampling.
        # output_padding used where needed to recover exact spatial dims.
        self.decoder = nn.Sequential(
            # Block 4 inverse: 256 -> 128,  (20,11) -> (40,22)
            nn.ConvTranspose2d(bc * 8, bc * 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(bc * 4),
            nn.ReLU(inplace=True),

            # Block 3 inverse: 128 -> 64,  (40,22) -> (80,44) -> pad to (80,45)
            nn.ConvTranspose2d(bc * 4, bc * 2, kernel_size=2, stride=2,
                               output_padding=(0, 1)),
            nn.BatchNorm2d(bc * 2),
            nn.ReLU(inplace=True),

            # Block 2 inverse: 64 -> 32,  (80,45) -> (160,90)
            nn.ConvTranspose2d(bc * 2, bc, kernel_size=2, stride=2),
            nn.BatchNorm2d(bc),
            nn.ReLU(inplace=True),

            # Block 1 inverse: 32 -> 1,  (160,90) -> (320,180)
            nn.ConvTranspose2d(bc, 1, kernel_size=2, stride=2),
            # No activation — output range matches normalised input
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, F) normalised spectrogram  (T=321, F=180)
        Returns:
            reconstruction: (B, T, F)
            latent: (B, 256, 20, 11) spatial bottleneck map
        """
        # Pad time dim from 321 to 322 (nearest multiple of 16 above 321
        # is 336, but we only need 322 to survive 4x pool-of-2: 322->161
        # ... that's odd.  Actually 320 is 2^4 * 20.  So pad to 320+2=322?
        # Let's just pad to 322 on input side, trim on output.)
        # Simpler: zero-pad 1 frame at the end so T becomes even at every stage.
        # 321 + 1 = 322 -> 161 (odd!) -> need 162 -> 81 (odd!) -> ...
        # Cleanest: pad to 320 by trimming 1 frame, or pad to 336.
        # Going with: keep 321, pad 1 -> 322 is still problematic.
        #
        # Actually the AvgPool2d(2) on odd dims: torch floors:
        # 321 -> 160, 160 -> 80, 80 -> 40, 40 -> 20.  That works!
        # 180 -> 90, 90 -> 45, 45 -> 22, 22 -> 11.    That works!
        # So no explicit padding needed.  The ConvTranspose chain
        # produces: 20->40, 40->80, 80->160, 160->320.
        # We need 321, so we pad the decoder output by 1 in time dim.

        x_4d = x.unsqueeze(1)                     # (B, 1, 321, 180)

        latent = self.encoder(x_4d)                # (B, 256, 20, 11)

        recon_4d = self.decoder(latent)            # (B, 1, 320, 180)

        # Trim or pad time dim to match original
        T_orig = x.size(1)
        T_recon = recon_4d.size(2)
        if T_recon < T_orig:
            # Pad with zeros at the end
            pad_t = T_orig - T_recon
            recon_4d = nn.functional.pad(recon_4d, (0, 0, 0, pad_t))
        elif T_recon > T_orig:
            recon_4d = recon_4d[:, :, :T_orig, :]

        reconstruction = recon_4d.squeeze(1)       # (B, 321, 180)

        return reconstruction, latent


if __name__ == "__main__":
    model = ConvAutoencoder()
    x = torch.randn(4, 321, 180)
    recon, latent = model(x)
    print(f"Input shape:          {x.shape}")
    print(f"Latent shape:         {latent.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    assert x.shape == recon.shape, f"Shape mismatch: {x.shape} vs {recon.shape}"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters:     {n_params:,}")
    print("Shape test passed.")
