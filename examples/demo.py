#!/usr/bin/env python3
"""
This file is not runnable, it shows an example of how you can 
structure your code for training, evaluation, and prediction.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scripts.evaluation import calculate_eer

class AudioDataset(Dataset):
    def __init__(self, features_path, labels_path):
        # load the features and labels from the paths
        # self.features_df = ...
        # self.labels_df = ...
        raise NotImplementedError("This method is not implemented, please implement it yourself")
    
    def __len__(self):
        raise NotImplementedError("This method is not implemented, please implement it yourself")
    
    def __getitem__(self, idx):
        # use idx to get the idx-th row of the features and labels
        # then return the features and label as tensors
        raise NotImplementedError("This method is not implemented, please implement it yourself")

class PredictDataset(Dataset):
    """
    This dataset is used to load the test features
    It doesn't have labels, so we don't need to load them
    """
    def __init__(self, feature_path):
        self.features_df = pd.read_pickle(feature_path)
    
    def __len__(self):
        return len(self.features_df)
    
    def __getitem__(self, idx):
        return self.uttids[idx], self.features_df.iloc[idx]['features']

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("This model is not implemented, please implement it yourself")
    
    def forward(self, x):
        raise NotImplementedError("This model is not implemented, please implement it yourself")

def evaluate(model, dataloader, device) :
    # when evaluating the model, set the model to the evaluation mode
    model.eval()

    all_scores = []
    all_labels = []
    # torch.no_grad() means the following code will not compute the gradients
    with torch.no_grad():
        for features, labels in tqdm(dataloader):
            features = features.to(device)
            # your model to provide the scores for the features
            scores = model(features)
            all_scores.append(scores)
            all_labels.append(labels)
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels).to(device)
    eer, _ = calculate_eer(scores, labels)

    # after evaluating, set the model to the training mode
    model.train()
    return eer

def prediction(model, testloader, output_path='prediction.pkl'):
    model.eval()
    uttids = []
    predictions = []

    # torch.no_grad() means the following code will not compute the gradients
    with torch.no_grad():
        for uttid, features in tqdm(testloader):
            scores = model(features)
            # we append the scores to the predictions list
            predictions.extend(scores.cpu().tolist())
            uttids.extend(uttid)
    pd.DataFrame({'uttid': uttids, 'predictions': predictions}).to_pickle(output_path)
    print(f"Saved predictions to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--dataset-name", dest="dataset_name", type=str, default="data")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default='cpu')
    
    
    device = torch.device(args.device)
    
    train_dataset = AudioDataset(args.train_features_path, args.train_labels_path)
    dev_dataset = AudioDataset(args.dev_features_path, args.dev_labels_path)
    test_dataset = PredictDataset(args.test_features_path)

    # prepare the data loaders for training, validation and testing
    # train_loader = ... 
    # dev_loader = ...
    # test_loader = ...
    
    model = Model().to(device)
    # example loss: we use BCEWithLogitsLoss for binary classification
    # of course you can use others
    criterion = nn.BCEWithLogitsLoss()
    # prepare the optimizer
    # optimizer = ...
    
    for epoch in tqdm(range(args.num_epochs), desc="train"):
        total_loss = 0
        for features, labels in tqdm(train_loader, desc=f"epoch {epoch+1}"):
            # your training code here
            pass
        
        dev_eer = evaluate(model, dev_loader, device)
        print(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {total_loss/len(train_loader):.4f}, Dev EER: {dev_eer:.4f}")
    
    prediction(model, test_loader, 'prediction.pkl')

if __name__ == "__main__":
    main()
