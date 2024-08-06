import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

def compute_multi_task_loss(variety_preds, variety, rating_preds, rating, 
                            classification_weight=1.0, regression_weight=1.0):
    
    # print('\n variety',variety)
    # Classification Loss
    classification_loss = torch.nn.CrossEntropyLoss()(variety_preds, variety.squeeze(-1))
    
    # Regression Loss
    if rating_preds.dim() > 1:
        rating_preds = rating_preds.squeeze(-1)  # Remove the last dimension if it's size 1
    
    rating = rating.float()  # Ensure rating is float
    regression_loss = torch.nn.MSELoss()(rating_preds, rating)
    
    # Weighted Average of Both Losses
    total_loss = classification_weight * classification_loss + \
                 regression_weight * regression_loss
    
    return total_loss

def train_model(model, train_dataset, eval_dataset, tokenizer, collate_fn, epochs=50, batch_size=8, lr=5e-3):
    # Prepare data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # Initialize loss function (assuming classification)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            variety = batch['variety']
            rating = batch['rating']           

            # Forward pass
            variety_preds, rating_preds = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Compute loss
            loss = compute_multi_task_loss(variety_preds, variety, rating_preds, rating)
            # print('loss', loss)
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        
        # Evaluate the model
        eval_loss, eval_accuracy = evaluate_model(model, eval_loader)
        print(f"Validation Loss: {eval_loss:.4f}, Accuracy: {eval_accuracy:.4f}")


def evaluate_model(model, eval_loader):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            variety = batch['variety']
            rating = batch['rating']
            
            # Forward pass
            variety_preds, rating_preds = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Compute loss
            loss = compute_multi_task_loss(variety_preds, variety, rating_preds, rating)
            
            total_loss += loss.item()
            
            # Compute accuracy (for classification)
            _, predicted = torch.max(variety_preds, dim=1)
            correct_predictions += (predicted == variety).sum().item()
            total_samples += variety.size(0)
    
    avg_loss = total_loss / len(eval_loader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy