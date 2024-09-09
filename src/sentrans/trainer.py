# trainer.py

from transformers import Trainer
from torch.utils.data import DataLoader 
from sentrans.constants import CLASSIFICATION_WT, REGRESSION_WT
import torch 
from .scaler_manager import ScalerManager

# Define a custom trainer
class SentenceTrainer(Trainer):    
    def compute_loss(self, model, inputs):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        age = inputs['age']
        rating = inputs['rating'] 
        age_preds, rating_preds = model(input_ids, attention_mask)
        classification_loss = torch.nn.CrossEntropyLoss()(rating_preds, rating)
          
        print('\n age_pred',age_preds)
        print('age',age)
        if age_preds.dim() > 1:
            age_preds = age_preds.squeeze(-1)  # Squeeze the last dimension if it's size 1

        # Ensure rating is of type float and the same shape
        age = age.float() 
        
        regression_loss = torch.nn.MSELoss()(age_preds, age)
        
        # Get weighted average of both the losses
        loss = CLASSIFICATION_WT * classification_loss + \
                REGRESSION_WT * regression_loss

        return loss
    
    def prediction_step(self, model, inputs, scaler,prediction_loss_only=False,ignore_keys=None): 
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']  
        
        with torch.no_grad():
            logits = [] 
            age_preds, rating_preds = model(input_ids=input_ids, attention_mask=attention_mask)
        
        scaler = ScalerManager.get_instance().get_scaler()
        age_preds = age_preds.detach().cpu().numpy()
        age_preds = scaler.inverse_transform(age_preds)
        age_preds = torch.tensor(age_preds, dtype=torch.float32)
        logits = (age_preds, rating_preds)

        if prediction_loss_only:
            return (None, logits, None)
 
        labels = (inputs['age'], inputs['rating'])
        return (None, logits, labels)

