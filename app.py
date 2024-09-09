from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from sentrans.tokenize import Tokenize
from sentrans.embeddings import Embeddings
from sentrans.dataset import WineDataset
from sentrans.constants import MODEL_NAME, TOKEN_MAX_LENGTH, DATA_FILE_PATH
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
mtl_model = None 

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_training')
def start_training():
    global st_model,mtl_model
    response = {
        "message": "Training started...",
        "status": "running"
    }
    emit('training_status', response)

    tokenize.prepare_dataset(DATA_FILE_PATH)
    tokenize.tokenize_data(tokenize.tokenizer, TOKEN_MAX_LENGTH)

    train_dataset = WineDataset(tokenize.tokenized_train_data, tokenize.train_df['age'].tolist(),\
                                tokenize.train_df['rating'].tolist())
    eval_dataset = WineDataset(tokenize.tokenized_eval_data, tokenize.eval_df['age'].tolist(), \
                               tokenize.eval_df['rating'].tolist())
    data_collator = SentenceDataCollator()
    mtl_model = MultiTaskBERT(model_name=MODEL_NAME,
                              num_classes=tokenize.num_classes) 
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
    # emit('final_accuracy', metrics['test_variety_accuracy'])
    emit(f"Variety Accuracy:{metrics['test_variety_accuracy']:.2f}")
    emit(f"Rating Accuracy:{metrics['test_rating_accuracy']:.2f}")
     

@app.route('/predict', methods=['POST'])
def predict():
    sentence = request.json['sentence']
    variety, rating = predict_sentence(mtl_model, tokenize.tokenizer, sentence, TOKEN_MAX_LENGTH)
    response = {
        "sentence": sentence,
        "predicted_variety": tokenize.class_label_decode[variety],
        "predicted_rating": rating
    }
    return jsonify(response)

@app.route('/random_sentence', methods=['GET'])
def random_sentence():
    with open('sample_wine_descriptions.txt', 'r') as file:
        sentences = file.readlines()
    random_sentence = random.choice(sentences)
    return jsonify({"sentence": random_sentence})

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=5000)
