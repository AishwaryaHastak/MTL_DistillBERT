from sentrans.dataset import NewsDataset
import torch
import pandas as pd

def predict_sentence(trainer, tokenizer, sentence, TOKEN_MAX_LENGTH):
    # Create a DataFrame for the input sentence
    pred_df = pd.DataFrame({'sentence': [sentence, sentence], 'label': [0, 0], 'task': ['class', 'sent']})
    
    # Tokenize the input sentence
    inputs = tokenizer(pred_df['sentence'].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=TOKEN_MAX_LENGTH)
    
    # Create dataset object for the encodings
    pred_dataset = NewsDataset(inputs, pred_df.label.tolist(), pred_df.task.tolist())
    
    # Get predictions
    predictions, labels, metrics = trainer.predict(pred_dataset)
    
    # Get the predicted class and sentiment
    predicted_class = torch.argmax(torch.tensor(predictions[0]), dim=-1).item()
    predicted_sentiment = torch.argmax(torch.tensor(predictions[1]), dim=-1).item()
    
    return predicted_class, predicted_sentiment