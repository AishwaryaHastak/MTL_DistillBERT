# constants.py

# Training Arguments
TRAINING_ARGUMENTS = {
    'output_dir': 'results',             # Output directory
    'num_train_epochs': 3,               # Large number of epochs for small dataset
    'per_device_train_batch_size': 1,    # batch size 1
    'per_device_eval_batch_size': 1,     # batch size 1
    'warmup_steps': 10,                  # Smaller warmup steps for small dataset
    'weight_decay': 0.01,                # Weight decay
    'logging_dir': 'logs',               # Directory for storing logs
    'logging_steps': 50,                 # Log every 50 steps
    'eval_strategy': 'steps',            # Evaluation strategy
    'eval_steps': 50,                    # Evaluate every 25 steps
    'save_steps': 100,                   # Save every 100 steps
    'save_total_limit': 1                # Keep only the last saved model
}

# Model and Tokenization Constants
TOKEN_MAX_LENGTH = 32
BATCH_SIZE = 4
MODEL_NAME = 'distilbert-base-uncased'

# File Paths
CLASSIFICATION_FILE_PATH = 'data/classification_data.csv'
SENTIMENT_FILE_PATH = 'data/sentiment_data.csv'
