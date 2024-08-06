import torch
import pandas as pd

def predict_sentence(model, tokenizer, sentence, TOKEN_MAX_LENGTH):
    inputs = tokenizer(sentence,padding=True,truncation=True,max_length=TOKEN_MAX_LENGTH, return_tensors='pt')
     
    model.eval() 
    with torch.no_grad():
        # Perform prediction
        pred_variety, pred_rating = model(**inputs)
    print(pred_variety.argmax(-1).item(), pred_rating.item())

    return pred_variety.argmax(-1).item(), pred_rating.item()