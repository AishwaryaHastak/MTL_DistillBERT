# tokenize.py

import pandas as pd 
from transformers import DistilBertTokenizer  
from .scaler_manager import ScalerManager
import torch
    
class Tokenize:
    def __init__(self, model_name):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.tokenized_train_data = None
        self.tokenized_eval_data = None
        self.class_weights = None

    def prepare_dataset(self, data_file_path): 
        reviews_df = pd.read_csv(data_file_path, index_col=0)
        reviews_df = reviews_df.sample(n=1000, random_state=42).reset_index(drop=True)
        unique_classes = sorted(pd.factorize(reviews_df['rating'])[1])
        self.num_classes = len(reviews_df['rating'].unique()) 
        
        # Create a dictionary mapping numerical values to original labels
        self.class_label_decode = {index: label for index, label in enumerate(unique_classes)}  
        
        # Adjust the rating values from 1-5 to 0-4
        reviews_df['rating'] = reviews_df['rating'] - 1

        # Initialize and fit scaler
        scaler = ScalerManager.get_instance().get_scaler()
        reviews_df['age'] =scaler.fit_transform(reviews_df[['age']]) 
        
        train_df = reviews_df.sample(frac=0.9, random_state=42)
        eval_df = reviews_df.drop(train_df.index)
        
        # Ensure the columns are correctly formatted
        self.train_df = train_df[['age', 'review', 'rating']]
        self.eval_df = eval_df[['age', 'review', 'rating']]

        # Calucalte class weights for accuracte loss calculation
        self.calc_class_wts(reviews_df)

    def tokenize_data(self, tokenizer,TOKEN_MAX_LENGTH): 
        # tokenize classification dataset
        self.tokenized_train_data = self.tokenizer(self.train_df['review'].tolist(), padding=True,\
                                               truncation=True,max_length=TOKEN_MAX_LENGTH, return_tensors='pt')
        
        # tokenize classification dataset
        self.tokenized_eval_data = self.tokenizer(self.eval_df['review'].tolist(), padding=True,\
                                              truncation=True,max_length=TOKEN_MAX_LENGTH, return_tensors='pt')
    
    def calc_class_wts(self,data_fd):

        # 1. Make the class_label decode be in order
        # 2. Get class weights as list of class labels 0 to 4
        # 3. Test again and check whether the regression loss is accurate
        #   Since the data is normalized and now has both negative and positive values, we need a loss function that is sensitve to sign
        # 4. Classification head should work correnclty once the class_wts are set

        class_counts = data_fd['rating'].value_counts() 
        max_count = class_counts.max() 
        class_weights = {class_label: round((max_count / count),2) for class_label, count in class_counts.items()}
        sorted(class_weights.items()) 
        self.class_weights = torch.tensor(list(class_weights.values()), dtype= torch.float) 
        