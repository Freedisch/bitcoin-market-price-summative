from flask import Flask, request, jsonify
import numpy as np
from src.model import load_keras_model
from src.prediction import preprocess_input, predict

app = Flask(__name__)

model = load_keras_model('models/model2.pkl')

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        data = request.get_json()
        input_data = np.array(data['prices'])
        processed_input = preprocess_input(input_data)
        
        prediction = predict(model, processed_input)

        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
