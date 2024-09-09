# metric.py

import numpy as np
from .scaler_manager import ScalerManager

def compute_metrics(pred):
    labels = pred.label_ids 
    age_preds, rating_preds = pred.predictions
    age, ratings = pred.label_ids 
    # print('rating_preds',rating_preds)
    # print('\n rating_preds, ratings',rating_preds.argmax(-1), ratings)

    rating_accuracy = (rating_preds.argmax(-1) == ratings).mean().item()
    rating_preds = rating_preds.flatten()
    ratings = ratings.flatten()

    scaler = ScalerManager.get_instance().get_scaler() 
    age_np = age.reshape(-1, 1)

    # Inverse transform to original scale 
    age_original = scaler.inverse_transform(age_np).flatten()
 
    # print('\n age_preds, age',age_preds, age_original)

    # Calculate Mean Absolute Error (MAE)
    age_errors = np.abs(age_preds - age_original)
    mae = np.mean(age_errors)

    # Calculate Mean Absolute Percentage Error (MAPE)
    # Avoid division by zero by adding a small constant to the denominator
    mape = np.mean(np.abs((age_errors / (age_original + 1e-8)) * 100))
    
    return {"rating_accuracy": rating_accuracy,
            "age_accuracy":100-mape}