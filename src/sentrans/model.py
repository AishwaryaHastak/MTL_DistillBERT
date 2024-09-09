# model.py

import torch
import torch.nn as nn
from transformers import DistilBertModel

class MultiTaskBERT(nn.Module):
    def __init__(self,model_name, num_classes):
        super(MultiTaskBERT, self).__init__()
        self.bert = DistilBertModel.from_pretrained(model_name) 
        self.dropout = nn.Dropout(0.6)
        self.classification = nn.Sequential(nn.Linear(self.bert.config.hidden_size, 32),
                                            nn.LeakyReLU(),
                                            nn.Linear(32, num_classes),
                                            nn.Softmax())
        
        # self.regression = nn.Linear(self.bert.config.hidden_size,1)
        self.regression = nn.Sequential(nn.Linear(self.bert.config.hidden_size, 32),
                                        nn.Linear(32,1)
                                        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # Take the [CLS] token representation
        pooled_output = self.dropout(pooled_output)
        
        age = self.regression(pooled_output)
        rating = self.classification(pooled_output)
        
        return age, rating
