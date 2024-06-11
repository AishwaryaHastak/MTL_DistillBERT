# collator.py

import torch

# Define the custom collator
class SentenceDataCollator:
    def __call__(self, batch):
        # print('batch', batch)
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.tensor([item['label'] for item in batch])#, dtype=torch.long)
        tasks = [item['task'] for item in batch]
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': labels,
            'task': tasks
        }
