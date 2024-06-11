# embeddings.py

import torch
from transformers import DistilBertModel

class Embeddings:
    def __init__(self, tokenizer, model_name):
        self.tokenizer = tokenizer
        self.model = DistilBertModel.from_pretrained(model_name)
        self.embeddings = None

    def create_embeddings(self, tokenized_data):
        input_ids = tokenized_data['input_ids']
        attention_mask = tokenized_data['attention_mask']
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        self.embeddings = outputs.last_hidden_state
