# metric.py

# Define metric
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Compute accuracy
    correct_predictions = (preds == labels).sum().item()
    total_predictions = len(labels)
    accuracy = correct_predictions / total_predictions
    
    return {"accuracy": accuracy}