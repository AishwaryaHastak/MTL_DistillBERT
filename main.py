# main.py

from src.sentrans.tokenize import Tokenize
from src.sentrans.embeddings import Embeddings
from src.sentrans.dataset import WineDataset
from src.sentrans.constants import MODEL_NAME, TOKEN_MAX_LENGTH, DATA_FILE_PATH 
from src.sentrans.model import MultiTaskBERT
from src.sentrans.collator import SentenceDataCollator
from src.sentrans.metric import compute_metrics
from src.sentrans.training_args import TrainingArgs
from src.sentrans.trainer import SentenceTrainer
from src.sentrans.predict import predict_sentence
from src.sentrans.train import train_model

from torch.utils.data import DataLoader

if __name__ == "__main__":
    # Initialize tokenizer and dataset 
    tokenize = Tokenize(MODEL_NAME)

    print('\n [DEBUG] Collecting Data and Tokenizing... \n')

    # Prepare and tokenize the dataset
    tokenize.prepare_dataset(DATA_FILE_PATH)
    tokenize.tokenize_data(tokenize.tokenizer, TOKEN_MAX_LENGTH)  

    print('\n [DEBUG] Creating Datasets... \n')
    # Create objects of Dataset class
    train_dataset = WineDataset(tokenize.tokenized_train_data, tokenize.train_df['age'].tolist(),\
                                tokenize.train_df['rating'].tolist())
    eval_dataset = WineDataset(tokenize.tokenized_eval_data, tokenize.eval_df['age'].tolist(), \
                               tokenize.eval_df['rating'].tolist())
    
    data_collator = SentenceDataCollator() 

    print('\n [DEBUG] Initializing the Mutli Task model... \n')
    # Initialize the Mutli Task model
    mtl_model = MultiTaskBERT(model_name=MODEL_NAME,
                              num_classes=tokenize.num_classes) 
     
    # Train the model
    train_model(mtl_model, train_dataset, eval_dataset, data_collator, class_weights=tokenize.class_weights)


    # # Initialize the Training Aeguments
    # training_args = TrainingArgs()

    # # Initialize the Trainer
    # trainer = SentenceTrainer(
    #     model=mtl_model,
    #     args=training_args.training_arguments,
    #     train_dataset=train_dataset, 
    #     eval_dataset=eval_dataset,      
    #     data_collator=data_collator,
    #     compute_metrics=compute_metrics
    # )
    
    # print('\n Training the Multi Task Model ... \n')
    # # Train the model
    # trainer.train()
    # print('\n Model has been trained successfully ... \n')

    # # Evaluate the model on the evaluaation dataset
    # predictions, labels, metrics = trainer.predict(eval_dataset)

    # print('\nEvaluating the Model ... \n')
    # print(metrics)
    # print(f"\nThe model accuracy is: \
    #       \n Rating Accuracy :{metrics['test_rating_accuracy']:.2f}, \
    #       Age Accuracy: :{metrics['test_age_accuracy']:.2f} ")

    # # Ask the user if they want to try some sentences
    # while True:
    #     user_input = input("\n\nDo you want to try out some wine descriptions? (y/n): ").strip().lower()
    #     if user_input == 'n':
    #         print("\nExiting...")
    #         break
    #     elif user_input == 'y':
    #         sentence = input("\nEnter your descriptions: ").strip()
    #         age,rating = predict_sentence(mtl_model, tokenize.tokenizer, sentence, TOKEN_MAX_LENGTH)
    #         print(f"\nPredicted Variety: {tokenize.class_label_decode[age]},\
    #                Predicted Rating: {rating}")
    #     else:
    #         print("\nInvalid input. Please enter 'y' or 'n'.")
    
     