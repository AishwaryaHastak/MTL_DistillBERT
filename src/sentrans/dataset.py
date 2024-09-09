# dataset.py


import torch
from torch.utils.data import Dataset

class WineDataset(Dataset):
    def __init__(self, encodings,age=None, rating=None): 
        self.encodings = encodings
        self.age = age
        self.rating = rating  

    def __len__(self):
        return len(self.age)

    def __getitem__(self, idx):  
        item = {key: val[idx] for key, val in self.encodings.items()}
        # item['variety']= torch.tensor(self.variety[idx], dtype=torch.long),
        item['label'] = {
            'age': torch.tensor(self.age[idx], dtype=torch.float64),
            'rating': torch.tensor(self.rating[idx], dtype=torch.long)
        }  
        return item
