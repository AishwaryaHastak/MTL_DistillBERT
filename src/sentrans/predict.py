from sentrans.dataset import NewsDataset
import torch
import pandas as pd

def predict_sentence(trainer, tokenizer, sentence, TOKEN_MAX_LENGTH):
    inputs = tokenizer(sentence)
    # Get predictions
    predictions, labels, metrics = trainer.predict(inputs) 
    # Convert predictions to tensors
    pred_variety, pred_rating = map(torch.tensor, predictions)


    # Get the predicted class and sentiment
    pred_variety = torch.argmax(pred_variety, dim=-1).item()
    # predicted_sentiment = torch.argmax(pred_sentiments, dim=-1).item()

    
    return pred_variety, pred_rating