import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_pickle_model(model_path):
    """
    Load a pickled model from a given file path.
    """
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def preprocess_input(input_data, scaler):
    """
    Preprocess the input data (e.g., scaling) similarly to the training phase.
    """
    scaled_data = scaler.transform(input_data.reshape(-1, 1))
    return scaled_data.reshape(1, -1)

def predict(model, input_data):
    """
    Make a prediction based on the processed input data.
    """
    prediction = model.predict(input_data)
    return prediction

if __name__ == "__main__":
    model = load_pickle_model('models/model4.pkl')

    input_data = np.array([1234.56, 1236.78, 1235.89, 1237.00, 1238.12])

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(input_data.reshape(-1, 1))
    processed_input = preprocess_input(input_data, scaler)

    prediction_output = predict(model, processed_input)
    print("Predicted Output:", prediction_output)
