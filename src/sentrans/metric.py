# metric.py

import numpy as np

def compute_metrics(pred):
    labels = pred.label_ids 
    variety_preds, rating_preds = pred.predictions
    variety, ratings = pred.label_ids 
    
    print('Variety acciracy', variety_preds, variety)
    variety_accuracy = (variety_preds.argmax(-1) == variety).mean().item()
    
    rating_errors = np.abs(rating_preds - ratings)
    rating_accuracy = 100 - (rating_errors.mean() / 100) 
    print(rating_errors, rating_accuracy)
    
    return {"variety_accuracy": variety_accuracy,
            "rating_accuracy":rating_accuracy}