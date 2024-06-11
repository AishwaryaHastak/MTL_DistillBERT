# dataset.py


import torch
from torch.utils.data import Dataset

class NewsDataset(Dataset):
    def __init__(self, encodings, labels,task):
        self.encodings = encodings
        self.labels = labels
        self.tasks = task

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx): 
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        item['task'] = self.tasks[idx]#, dtype=torch.str)
        return item
