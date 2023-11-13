"""
DevOps: Viral Gorecha

Basic Flask REST API
"""

import pickle
from pathlib import Path
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np

MAX_LENGTH = 18 # taken from the training notebook
CWD = Path().cwd()
MODEL_PATH = CWD / "artifacts" / "siamese_model_saved_artifacts"
MODEL_WEIGHTS_PATH = CWD / "artifacts" / "siamese_LSTM_embedding.h5"
TOKENIZER_PATH = CWD / "artifacts" / "tokenizer.pickle"

app = Flask(__name__)

# function to load tokenizer
def load_tokenizer():
    with open(str(TOKENIZER_PATH), 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer


def preprocess_inputs(data):
    # Tokenize and preprocess input data
    tokenized_entity_1 = tokenizer.texts_to_sequences(data['entity_1'])
    tokenized_entity_2 = tokenizer.text_to_sequencies(data['entity_2'])
     
    # Pad or truncate sequences to the same length used during training
    padded_entity_1 = pad_sequences(tokenized_entity_1, maxlen=MAX_LENGTH)
    padded_entity_2 = pad_sequences(tokenized_entity_2, maxlen=MAX_LENGTH) 

    return padded_entity_1, padded_entity_2


# Load the pre-trained Siamese model
siamese_model = load_model(str(MODEL_PATH)) 

# load weights in case we change the arch where model can be loaded from a separate python file
# siamese_model.load_weights(str(MODEL_WEIGHTS_PATH))

# load the same tokenizer that was used during training
tokenizer = load_tokenizer()


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # preprocess inputs
    padded_entity_1, padded_entity_2 = preprocess_inputs(data)

    # Make predictions
    predictions = siamese_model.predict([padded_entity_1, padded_entity_2])

    # process prediction results
    prediction_result = {'is_match': bool(predictions[0] > 0.5), 'confidence': float(predictions[0])}

    return jsonify(prediction_result)

if __name__ == '__main__':
    app.run(debug=True)
