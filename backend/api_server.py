from flask import Flask, request, jsonify
from main import load_model, load_optimizer, ModelRunner

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train_model():
    pass

@app.route('/predict', methods=['POST'])
def make_predictions():
    pass

@app.route('/models', methods=['GET'])
def show_models():
    pass

@app.route('make_predictions')
def make_predictions():
    pass