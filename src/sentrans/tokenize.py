# tokenize.py

import pandas as pd 
from transformers import DistilBertTokenizer 
    
class Tokenize:
    def __init__(self, model_name):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.tokenized_train_data = None
        self.tokenized_eval_data = None

    def prepare_dataset(self, data_file_path): 

        wine_df = pd.read_csv(data_file_path)
        wine_df = wine_df.sample(n=5000, random_state=42).reset_index(drop=True)
        wine_df['variety'], unique_classes = pd.factorize(wine_df['variety']) 
        self.num_classes = len(wine_df['variety'].unique()) 
        
        # Create a dictionary mapping numerical values to original labels
        self.class_label_decode = {index: label for index, label in enumerate(unique_classes)} 
        
        train_df = wine_df.sample(frac=0.8, random_state=42)
        eval_df = wine_df.drop(train_df.index)
        
        # Ensure the columns are correctly formatted
        self.train_df = train_df[['variety', 'rating', 'notes']]
        self.eval_df = eval_df[['variety', 'rating', 'notes']]

    def tokenize_data(self, tokenizer,TOKEN_MAX_LENGTH): 
        # tokenize classification dataset
        self.tokenized_train_data = self.tokenizer(self.train_df['notes'].tolist(), padding=True,\
                                               truncation=True,max_length=TOKEN_MAX_LENGTH, return_tensors='pt')
        
        # tokenize classification dataset
        self.tokenized_eval_data = self.tokenizer(self.eval_df['notes'].tolist(), padding=True,\
                                              truncation=True,max_length=TOKEN_MAX_LENGTH, return_tensors='pt')