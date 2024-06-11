# model.py

import torch
import torch.nn as nn
from transformers import DistilBertModel

class MultiTaskBERT(nn.Module):
    def __init__(self, num_classes_topic, num_classes_sentiment,model):
        super(MultiTaskBERT, self).__init__()
        # self.bert = DistilBertModel.from_pretrained(MODEL_NAME)
        self.bert = model
        self.dropout = nn.Dropout(0.1)
        self.classifier_topic = nn.Linear(self.bert.config.hidden_size, num_classes_topic)
        self.classifier_sentiment = nn.Linear(self.bert.config.hidden_size, num_classes_sentiment) 

    def forward(self, input_ids, attention_mask, task=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # Take the [CLS] token representation
        pooled_output = self.dropout(pooled_output)
        if 'class' in task:
            return self.classifier_topic(pooled_output)
        elif 'sent' in task:
            return self.classifier_sentiment(pooled_output)
        else:
            return None
