# dataset.py


import torch
from torch.utils.data import Dataset

class NewsDataset(Dataset):
    def __init__(self, encodings,variety, rating):#,topics, sentiments):
        self.encodings = encodings
        self.variety = variety
        self.rating = rating  

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx): 
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['label'] = {
            'variety': torch.tensor(self.variety[idx], dtype=torch.long),
            'rating': torch.tensor(self.rating[idx], dtype=torch.long)
        }  
        return item
