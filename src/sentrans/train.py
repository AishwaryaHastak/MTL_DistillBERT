import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
from sentrans.constants import CLASSIFICATION_WT, REGRESSION_WT
from .scaler_manager import ScalerManager

def compute_multi_task_loss(age_preds, age, rating_preds, rating, class_weights=None):
    # Classification Loss
    classification_loss = torch.nn.CrossEntropyLoss(weight= class_weights)(rating_preds, rating)
    # print('\n rating_preds, rating', rating_preds, rating)
    # print('\n classification_loss', classification_loss)

    # Regression Loss
    if age_preds.dim() > 1:
        age_preds = age_preds.squeeze(-1)  # Squeeze the last dimension if it's size 1

    # # Ensure rating is of type float and the same shape
    age = age.float() 
    
    # regression_loss = torch.nn.MSELoss()(age_preds, age)
    regression_loss = torch.nn.SmoothL1Loss()(age_preds, age)
    # print('\n age_preds, age', age_preds, age)
    # print('\n regression_loss', regression_loss)
    
    # Get weighted average of both the losses
    loss = CLASSIFICATION_WT * classification_loss + \
            REGRESSION_WT * regression_loss

    return loss

def train_model(model, train_dataset, eval_dataset, collate_fn, class_weights, epochs=3, batch_size=8, lr=5e-5):
    # Prepare data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=lr) 
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            age = batch['age']
            rating = batch['rating']           

            # Forward pass
            age_preds, rating_preds = model(input_ids=input_ids, attention_mask=attention_mask)
            # print('\n age_preds, rating_preds',age_preds, rating_preds)
            # print('\n age, rating',age, rating)
            # Compute loss
            loss = compute_multi_task_loss(age_preds, age, rating_preds, rating, class_weights)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        print(f"\n Average Training Loss: {avg_train_loss:.4f}")
        
        # Evaluate the model
        eval_loss, eval_accuracy, reg_accuracy = evaluate_model(model, eval_loader)
        print(f"\n Validation Loss: {eval_loss:.4f}, \
              Classification Accuracy: {eval_accuracy:.4f}, \
              Regression Accuracy: {reg_accuracy:.4f}")


def evaluate_model(model, eval_loader):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    mape_acc = 0
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            age = batch['age']
            rating = batch['rating']

            # print('\ninput age:',age)
            # Forward pass
            age_preds, rating_preds = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # print('\nage_preds, rating_preds ',age_preds, rating_preds )
            # Compute loss
            loss = compute_multi_task_loss(age_preds, age, rating_preds, rating)
            
            total_loss += loss.item()
            
            # Compute accuracy (for classification)
            _, predicted = torch.max(age_preds, dim=1)
            correct_predictions += (predicted == rating).sum().item()
            total_samples += rating.size(0)
            scaler = ScalerManager.get_instance().get_scaler() 
            age_preds = age_preds.detach().cpu().numpy()
            age_preds = scaler.inverse_transform(age_preds)
            age_np = age.reshape(-1, 1)

            # Inverse transform to original scale 
            age_original = scaler.inverse_transform(age_np).flatten()
        
            # print('\n age_preds, age',age_preds, age_original)
            # print('\n rating_preds, rating',rating_preds, rating)

            # Calculate Mean Absolute Error (MAE)
            age_errors = np.abs(age_preds - age_original)
            mae = np.mean(age_errors)

            # Calculate Mean Absolute Percentage Error (MAPE)
            # Avoid division by zero by adding a small constant to the denominator
            mape = np.mean(np.abs((age_errors / (age_original + 1e-8)) * 100))
            mape_acc += mape
    
    avg_loss = total_loss / len(eval_loader)
    accuracy = correct_predictions / total_samples
    regression_acc = mape_acc/len(eval_loader)
    return avg_loss, accuracy, regression_acc