# trainer.py

from transformers import Trainer
from torch.utils.data import DataLoader
from sentrans.constants import CLASSIFICATION_WT, REGRESSION_WT
import torch 

# Define a custom trainer
class SentenceTrainer(Trainer):    
    def compute_loss(self, model, inputs):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        variety = inputs['variety']
        rating = inputs['rating'] 
        variety_preds, rating_preds = model(input_ids, attention_mask)
        classification_loss = torch.nn.CrossEntropyLoss()(variety_preds, variety)
         
        if rating_preds.dim() > 1:
            rating_preds = rating_preds.squeeze(-1)  # Squeeze the last dimension if it's size 1

        # Ensure rating is of type float and the same shape
        rating = rating.float()

        regression_loss = torch.nn.MSELoss()(rating_preds, rating)
        
        # Get weighted average of both the losses
        loss = CLASSIFICATION_WT * classification_loss + \
                REGRESSION_WT * regression_loss

        return loss
    
    def prediction_step(self, model, inputs, prediction_loss_only,ignore_keys=None): 
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']  
        # We do not need to calculate gradients for validation
        with torch.no_grad():
            logits = [] 
            pred_topics, pred_sentiments = model(input_ids=input_ids, attention_mask=attention_mask)
 
        logits = (pred_topics, pred_sentiments)

        if prediction_loss_only:
            return (None, logits, None)
 
        labels = (inputs['variety'], inputs['rating'])
        return (None, logits, labels)

