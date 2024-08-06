# metric.py

import numpy as np

def compute_metrics(pred):
    labels = pred.label_ids 
    variety_preds, rating_preds = pred.predictions
    variety, ratings = pred.label_ids 
     
    print('\n variety_preds, variety',variety_preds.argmax(-1), variety)

    variety_accuracy = (variety_preds.argmax(-1) == variety).mean().item()
    rating_preds = rating_preds.flatten()
    ratings = ratings.flatten()

    # Calculate Mean Absolute Error (MAE)
    rating_errors = np.abs(rating_preds - ratings)
    mae = np.mean(rating_errors)

    # Calculate Mean Absolute Percentage Error (MAPE)
    # Avoid division by zero by adding a small constant to the denominator
    mape = np.mean(np.abs((rating_errors / (ratings + 1e-8)) * 100))
    
    return {"variety_accuracy": variety_accuracy,
            "rating_accuracy":100-mape}