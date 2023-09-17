from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents from the JSON file
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Initialize the neural network model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"
app = Flask(__name__)
CORS(app)
def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    return "Sorry 😣, I do not understand..."

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/api/chat', methods=['POST'])
def process_chat():
    try:
        # Get data from the request body as JSON
        data = request.get_json()
        if 'message' in data:
            user_message = data['message']
            response = get_response(user_message)
            return jsonify({"message": response})
        else:
            return jsonify({"message": "Invalid request format"}), 400

    except json.JSONDecodeError:
        return jsonify({"message": "Invalid JSON data"}), 400

if __name__ == '__main__':
    app.run(debug=True)
