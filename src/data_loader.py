```python
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class PTADataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class PTADataloader:
    def __init__(self, data_path, test_size=0.2, random_state=42):
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state

    def load_data(self):
        data = pd.read_csv(self.data_path)
        labels = data.pop('label')
        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=self.test_size, random_state=self.random_state)
        return train_data.values, test_data.values, train_labels.values, test_labels.values

    def get_dataloader(self, batch_size, shuffle=True):
        train_data, test_data, train_labels, test_labels = self.load_data()
        train_dataset = PTADataset(train_data, train_labels)
        test_dataset = PTADataset(test_data, test_labels)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
        return train_dataloader, test_dataloader
```