import os
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)
model_path = os.path.join(os.path.dirname(__file__), 'model', 'model.pkl')

# Load the model
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Expecting JSON payload
    features = data.get('features')  # Example: {"features": [0.1, 0.2, 0.3]}
    if not features:
        return jsonify({'error': 'No features provided'}), 400

    # Convert features to a numpy array and predict
    prediction = model.predict(np.array([features]))
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))  # Dynamic port for Render
    app.run(debug=True, port=port, host='0.0.0.0')
