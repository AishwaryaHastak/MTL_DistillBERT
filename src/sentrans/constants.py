# constants.py

# Training Arguments
TRAINING_ARGUMENTS = {
    'output_dir': 'results',             # Output directory
    'num_train_epochs': 2,               # Large number of epochs for small dataset
    'per_device_train_batch_size': 20,    # batch size 10
    'per_device_eval_batch_size': 20,     # batch size 10
    'warmup_steps': 0,                  # Smaller warmup steps for small dataset
    'weight_decay': 0.01,                # Weight decay
    'logging_dir': 'logs',               # Directory for storing logs
    'logging_steps': 40,                 # Log every 50 steps
    'eval_strategy': 'steps',            # Evaluation strategy
    'eval_steps': 50,                    # Evaluate every 25 steps
    'save_steps': 100,                   # Save every 100 steps
    'save_total_limit': 1,                # Keep only the last saved model
    'learning_rate':2e-2                # Learning Rate
}

# Model and Tokenization Constants
TOKEN_MAX_LENGTH = 32 
MODEL_NAME = 'distilbert-base-uncased'

# File Paths
# DATA_FILE_PATH = 'data/reviews.csv' 
DATA_FILE_PATH = 'data/wine-ratings.csv' 
MODEL_PATH = 'results/checkpoint-250' 

# Task Weights
CLASSIFICATION_WT = 0.7
REGRESSION_WT = 0.3