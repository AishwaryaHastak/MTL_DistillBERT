# model.py

import torch
import torch.nn as nn
from transformers import DistilBertModel

class MultiTaskBERT(nn.Module):
    def __init__(self,model, num_classes):
        super(MultiTaskBERT, self).__init__()
        
        self.bert = model
        self.dropout = nn.Dropout(0.1)
        self.classification = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.regression = nn.Linear(self.bert.config.hidden_size,1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # Take the [CLS] token representation
        pooled_output = self.dropout(pooled_output)
        variety = self.classification(pooled_output)
        rating = self.regression(pooled_output)
        return variety, rating
