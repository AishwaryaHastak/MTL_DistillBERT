# collator.py

import torch

# Define the custom collator
class SentenceDataCollator:
    def __call__(self, batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        variety = torch.tensor([item['label']['variety'] for item in batch], dtype=torch.long)
        rating = torch.tensor([item['label']['rating'] for item in batch], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'variety': variety,
            'rating': rating
        }

        
