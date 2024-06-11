# tokenize.py

import pandas as pd
# from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer 
    
class Tokenize:
    def __init__(self, model_name):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.tokenized_train_data = None
        self.tokenized_eval_data = None

    def prepare_dataset(self, classification_file_path, sentiment_file_path):
       # Load the topic classificaiton dataset
        classification_df = pd.read_csv(classification_file_path)
        classification_df['label'], unique_labels = pd.factorize(classification_df['label'])
        self.num_classes = len(classification_df.label.unique())

        # Create a dictionary mapping numerical values to original labels
        self.class_label_decode = {index: label for index, label in enumerate(unique_labels)}
        
        # Load the sentiment dataset
        sentiment_df = pd.read_csv(sentiment_file_path)
        # Transform text label to numerical label 
        sentiment_df['label'], unique_labels = pd.factorize(sentiment_df['label'])
        self.num_sentiment = len(unique_labels)

        # Create a dictionary mapping numerical values to original labels
        self.sent_label_decode = {index: label for index, label in enumerate(unique_labels)}
        
        classification_df['task'] = 'class'
        sentiment_df['task'] = 'sent'
        combined_df = pd.concat([classification_df, sentiment_df]).reset_index(drop=True)
        
        # Split the combined dataset
        # train_df, eval_df = train_test_split(combined_df, test_size=0.2, stratify=combined_df['task'])
        train_df = combined_df.sample(frac=0.8, random_state=42)
        eval_df = combined_df.drop(train_df.index)

        # Ensure the columns are correctly formatted
        self.train_df = train_df[['sentence', 'label', 'task']]
        self.eval_df = eval_df[['sentence', 'label', 'task']]

    def tokenize_data(self, tokenizer,TOKEN_MAX_LENGTH): 
        # tokenize classification dataset
        self.tokenized_train_data = self.tokenizer(self.train_df['sentence'].tolist(), padding=True,\
                                               truncation=True,max_length=TOKEN_MAX_LENGTH, return_tensors='pt')

        # tokenize classification dataset
        self.tokenized_eval_data = self.tokenizer(self.eval_df['sentence'].tolist(), padding=True,\
                                              truncation=True,max_length=TOKEN_MAX_LENGTH, return_tensors='pt')