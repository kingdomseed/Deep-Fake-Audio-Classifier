import pandas as pd
import torch
from torch.utils.data import Dataset


class AudioDeepfakeDataset(Dataset):
    """
    Dataset class for audio deepfake detection.
    Returns (features, label) pairs where:
        - features: torch.Tensor of shape (feature_dim, seq_len)
        - stored shape = [180, 321] (feature_dim=180, seq_len=321)
        - label: int (0=deepfake, 1=real)
    """

    def __init__(self, features_path, labels_path):
        """
        Load features.pkl and labels.pkl, merge them on 'uttid'

        Args:
            features_path: path to features.pkl
            labels_path: path to labels.pkl
        """
        # 1. Load features DataFrame
        self.features = pd.read_pickle(features_path)
        # 2. Load labels DataFrame
        self.labels = pd.read_pickle(labels_path)
        # 3. Merge on 'uttid' column (how='inner')
        self.data = pd.merge(self.features, self.labels, on="uttid", how="inner")
        # 4. Store the merged dataframe
        self.data = self.data.reset_index(drop=True)

    def __len__(self):
        """
        Return the number of samples
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return the idx-th sample as (features, label)

        Args:
            idx: Sample index

        Returns:
            features: torch.Tensor [180, 321]
            label: torch.Tensor scalar (or int)
        """
        # 1. Get the row at index idx
        row = self.data.iloc[idx]
        # 2. Extract features tensor
        features = row["features"].float()
        # 3. Extract label as integer
        label = torch.tensor(row["label"], dtype=torch.float32)
        # 4. Return (features, label)
        return features, label


if __name__ == "__main__":
    # Quick test
    dataset = AudioDeepfakeDataset("data/train/features.pkl", "data/train/labels.pkl")
    print(f"Dataset size: {len(dataset)}")

    features, label = dataset[0]
    print(f"Features shape: {features.shape}")  # Should be [180, 321]
    print(f"Features dtype: {features.dtype}")  # Should be torch.float32
    print(f"Label: {label}")  # Should be 0.0 or 1.0
    print(f"Label dtype: {label.dtype}")  # Should be torch.float32
