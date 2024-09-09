import torch
import pandas as pd
from .scaler_manager import ScalerManager

def predict_sentence(model, tokenizer, sentence, TOKEN_MAX_LENGTH):
    inputs = tokenizer(sentence,padding=True,truncation=True,max_length=TOKEN_MAX_LENGTH, return_tensors='pt')
     
    model.eval() 
    with torch.no_grad():
        # Perform prediction
        pred_age, pred_rating = model(**inputs)
    
    scaler = ScalerManager.get_instance().get_scaler()
    # Use inverse_transform or other methods 
    age_preds = age_preds.detach().cpu().numpy()
    age_preds = scaler.inverse_transform(age_preds)
    age_preds = torch.tensor(age_preds, dtype=torch.float32)

    print(pred_rating.argmax(-1).item(), pred_age.item())

    return pred_age.item(),pred_rating.argmax(-1).item()