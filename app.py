from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
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
import random

app = Flask(__name__)
socketio = SocketIO(app)

tokenize = Tokenize(MODEL_NAME)
st_model = None
trainer = None

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_training')
def start_training():
    global st_model, trainer
    response = {
        "message": "Training started...",
        "status": "running"
    }
    emit('training_status', response)

    tokenize.prepare_dataset(CLASSIFICATION_FILE_PATH, SENTIMENT_FILE_PATH)
    tokenize.tokenize_data(tokenize.tokenizer, TOKEN_MAX_LENGTH)
    st_model = Embeddings(tokenize.tokenizer, MODEL_NAME)
    st_model.create_embeddings(tokenize.tokenized_train_data)

    # Print embeddings for a few sentences
    for i in range(2):  # Adjust the range as needed
        print('\nSentence: \n', tokenize.train_df.iloc[i]['sentence'], '\nEmbedding: \n', st_model.embeddings[i])
        print('\nEmbedding Shape: \n', st_model.embeddings[i].shape)

    train_dataset = NewsDataset(tokenize.tokenized_train_data, tokenize.train_df['label'].tolist(), tokenize.train_df['task'].tolist())
    eval_dataset = NewsDataset(tokenize.tokenized_eval_data, tokenize.eval_df['label'].tolist(), tokenize.eval_df['task'].tolist())
    data_collator = SentenceDataCollator()
    mtl_model = MultiTaskBERT(num_classes_topic=tokenize.num_classes, num_classes_sentiment=tokenize.num_sentiment, model=st_model.model)
    training_args = TrainingArgs()

    class CustomTrainer(SentenceTrainer):
        def training_step(self, *args, **kwargs):
            step_output = super().training_step(*args, **kwargs)
            progress = self.state.global_step / self.state.max_steps
            if progress < 0.5:
                status_message = "Training started..."
            elif progress < 0.8:
                status_message = "Hang in there..."
            elif progress < 0.9:
                status_message = "Almost there!"
            else:
                status_message = "Adding some finishing touches..."
            emit('training_progress', {'progress': progress, 'status_message': status_message})
            return step_output

        def evaluation_step(self, *args, **kwargs):
            eval_output = super().evaluation_step(*args, **kwargs)
            progress = self.state.global_step / self.state.max_steps
            emit('evaluation_progress', {'progress': progress})
            return eval_output

    trainer = CustomTrainer(
        model=mtl_model,
        args=training_args.training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    response["message"] = "Model has been trained successfully."
    response["status"] = "completed"
    emit('training_status', response)

    # Emit final training progress
    emit('training_progress', {'progress': 1.0})

    predictions, labels, metrics = trainer.predict(eval_dataset)
    emit('final_accuracy', metrics['test_accuracy'])

@app.route('/predict', methods=['POST'])
def predict():
    sentence = request.json['sentence']
    topic, sentiment = predict_sentence(trainer, tokenize.tokenizer, sentence, TOKEN_MAX_LENGTH)
    response = {
        "sentence": sentence,
        "predicted_topic": tokenize.class_label_decode[topic],
        "predicted_sentiment": tokenize.sent_label_decode[sentiment]
    }
    return jsonify(response)

@app.route('/random_sentence', methods=['GET'])
def random_sentence():
    with open('sample_sentences.txt', 'r') as file:
        sentences = file.readlines()
    random_sentence = random.choice(sentences)
    return jsonify({"sentence": random_sentence})

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=5000)
