import pandas as pd
import torch

try:
    df = pd.read_pickle("data/train/features.pkl")
    print("Columns:", df.columns)
    first_feature = df.iloc[0]["features"]
    if isinstance(first_feature, torch.Tensor):
        print(f"Shape: {first_feature.shape}")
    else:
        print(f"Type: {type(first_feature)}")
        # If it's numpy or list, print shape/len
        try:
            print(f"Shape: {first_feature.shape}")
        except:
            print(f"Len: {len(first_feature)}")

except Exception as e:
    print(e)
