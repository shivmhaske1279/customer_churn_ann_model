import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

# Initialize Flask with template directory pointing to project root templates folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))

# Load model pickle file
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model failed to load on server.'}), 500

    try:
        data = request.get_json(silent=True)
        
        if not data or 'features' not in data:
            return jsonify({'error': 'Missing "features" in request body.'}), 400

        features = data['features']
        
        # Ensure input contains exactly 10 float features expected by your Keras model input layer
        if len(features) != 10:
            return jsonify({'error': f'Expected 10 input values, but received {len(features)}.'}), 400

        # Reshape for single batch prediction (shape: [1, 10])
        input_array = np.array([features], dtype=np.float32)
        
        # Predict using Keras model
        prediction = model.predict(input_array)
        probability = float(prediction[0][0])
        predicted_class = int(probability >= 0.5)

        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'probability': round(probability, 4)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
