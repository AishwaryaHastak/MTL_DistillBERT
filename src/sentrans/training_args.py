# training_args.py

from transformers import TrainingArguments
from sentrans.constants import TRAINING_ARGUMENTS

class TrainingArgs():
    def __init__(self):
        self.training_arguments = TrainingArguments(
            output_dir=TRAINING_ARGUMENTS['output_dir'],
            num_train_epochs=TRAINING_ARGUMENTS['num_train_epochs'],
            per_device_train_batch_size=TRAINING_ARGUMENTS['per_device_train_batch_size'],
            per_device_eval_batch_size=TRAINING_ARGUMENTS['per_device_eval_batch_size'],
            warmup_steps=TRAINING_ARGUMENTS['warmup_steps'],
            weight_decay=TRAINING_ARGUMENTS['weight_decay'],
            logging_dir=TRAINING_ARGUMENTS['logging_dir'],
            logging_steps=TRAINING_ARGUMENTS['logging_steps'],
            eval_strategy=TRAINING_ARGUMENTS['eval_strategy'],
            eval_steps=TRAINING_ARGUMENTS['eval_steps'],
            save_steps=TRAINING_ARGUMENTS['save_steps'],
            save_total_limit=TRAINING_ARGUMENTS['save_total_limit']
        )
