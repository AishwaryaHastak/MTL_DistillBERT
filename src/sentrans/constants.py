# constants.py

# Training Arguments
TRAINING_ARGUMENTS = {
    'output_dir': 'results',             # Output directory
    'num_train_epochs': 5,               # Large number of epochs for small dataset
    'per_device_train_batch_size': 4,    # batch size 1
    'per_device_eval_batch_size': 4,     # batch size 1
    'warmup_steps': 0,                  # Smaller warmup steps for small dataset
    'weight_decay': 0.01,                # Weight decay
    'logging_dir': 'logs',               # Directory for storing logs
    'logging_steps': 1,                 # Log every 50 steps
    'eval_strategy': 'steps',            # Evaluation strategy
    'eval_steps': 1,                    # Evaluate every 25 steps
    'save_steps': 1,                   # Save every 100 steps
    'save_total_limit': 1                # Keep only the last saved model
}

# Model and Tokenization Constants
TOKEN_MAX_LENGTH = 32
BATCH_SIZE = 4
MODEL_NAME = 'distilbert-base-uncased'

# File Paths
DATA_FILE_PATH = 'data/wine-ratings.csv'
# DATA_FILE_PATH = 'data/data.csv'

# Task Weights
CLASSIFICATION_WT = 0.5
REGRESSION_WT = 0.5