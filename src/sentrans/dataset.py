# dataset.py


import torch
from torch.utils.data import Dataset

class WineDataset(Dataset):
    def __init__(self, encodings,variety=None, rating=None): 
        self.encodings = encodings
        # print(self.encodings.items())
        # print(self.encodings.items()['input_ids'])
        # self.inputs_ids = encodings['inputs_ids']
        # self.attention_mask = encodings['attention_mask']
        self.variety = variety
        self.rating = rating  

    def __len__(self):
        return len(self.variety)

    def __getitem__(self, idx):  
        item = {key: val[idx] for key, val in self.encodings.items()}
        # item['variety']= torch.tensor(self.variety[idx], dtype=torch.long),
        item['label'] = {
            'variety': torch.tensor(self.variety[idx], dtype=torch.long),
            'rating': torch.tensor(self.rating[idx], dtype=torch.long)
        }  
        # item['variety']= torch.tensor(self.variety[idx], dtype=torch.long),
        # item['rating']= torch.tensor(self.rating[idx], dtype=torch.long)
        # print(item)
        return item
