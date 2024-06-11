# main.py

from sentrans.tokenize import Tokenize
from sentrans.embeddings import Embeddings
from sentrans.dataset import NewsDataset
from sentrans.constants import MODEL_NAME, TOKEN_MAX_LENGTH, CLASSIFICATION_FILE_PATH, SENTIMENT_FILE_PATH
from sentrans.model import MultiTaskBERT
from sentrans.collator import SentenceDataCollator
from sentrans.metric import compute_metrics
from sentrans.training_args import TrainingArgs
from sentrans.trainer import SentenceTrainer
from sentrans.predict import predict_sentence


if __name__ == "__main__":
    # Initialize tokenizer and dataset 
    tokenize = Tokenize(MODEL_NAME)

    print('\nCollecting Data and Tokenizing... \n')

    # Prepare and tokenize the dataset
    tokenize.prepare_dataset(CLASSIFICATION_FILE_PATH, SENTIMENT_FILE_PATH)
    tokenize.tokenize_data(tokenize.tokenizer, TOKEN_MAX_LENGTH) 
    print('\nGenerating Embeddings... \n')

    # Initialize model and create embeddings
    st_model = Embeddings(tokenize.tokenizer, MODEL_NAME)
    st_model.create_embeddings(tokenize.tokenized_train_data)
    
    # Print embeddings for a few sentences
    for i in range(0, 2):
        # print('\nSentence: \n', tokenize.train_df.sentence[i], '\nEmbedding: \n', st_model.embeddings[i])
        print('\nSentence: \n', tokenize.train_df.iloc[i]['sentence'], '\nEmbedding: \n', st_model.embeddings[i])
        print('\nEmbedding Shape: \n', st_model.embeddings[i].shape)


    # Create objects of Dataset class
    train_dataset = NewsDataset(tokenize.tokenized_train_data, tokenize.train_df['label'].tolist(),\
                                tokenize.train_df['task'].tolist())
    eval_dataset = NewsDataset(tokenize.tokenized_eval_data, tokenize.eval_df['label'].tolist(), \
                               tokenize.eval_df['task'].tolist())
    

    data_collator = SentenceDataCollator()

    # Initialize the Mutli Task model
    mtl_model = MultiTaskBERT(num_classes_topic=tokenize.num_classes, 
                              num_classes_sentiment=tokenize.num_sentiment,
                              model=st_model.model)

    # Initialize the Training Aeguments
    training_args = TrainingArgs()

    # Initialize the Trainer
    trainer = SentenceTrainer(
        model=mtl_model,
        args=training_args.training_arguments,
        train_dataset=train_dataset,   # Use the classification dataset for the initial example
        eval_dataset=eval_dataset,         # Use the sentiment dataset for evaluation
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )


    print('\n Training the Multi Task Model ... \n')
    # Train the model
    trainer.train()
    print('\n Model has been trained successfully ... \n')

    # Evaluate the model on the evaluaation dataset
    predictions, labels, metrics = trainer.predict(eval_dataset)

    print('\nEvaluating the Model ... \n')
    print(f"\nThe model accuracy is: {metrics['test_accuracy']:.2f}")

    # Ask the user if they want to try some sentences
    while True:
        user_input = input("\n\nDo you want to try out some sentences? (y/n): ").strip().lower()
        if user_input == 'n':
            print("\nExiting...")
            break
        elif user_input == 'y':
            sentence = input("\nEnter your sentence: ").strip()
            topic,sentiment = predict_sentence(trainer, tokenize.tokenizer, sentence, TOKEN_MAX_LENGTH)
            print(f"\nPredicted Topic: {tokenize.class_label_decode[topic]},\
                   Predicted Sentiment: {tokenize.sent_label_decode[sentiment]}")
        else:
            print("\nInvalid input. Please enter 'y' or 'n'.")
    