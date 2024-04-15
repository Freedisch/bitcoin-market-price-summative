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
    # Assuming input_data is already formatted as a numpy array
    scaled_data = scaler.transform(input_data.reshape(-1, 1))
    # You might need to reshape the data depending on your model's input requirement
    return scaled_data.reshape(1, -1)  # Example reshape for a single sample with multiple features

def predict(model, input_data):
    """
    Make a prediction based on the processed input data.
    """
    prediction = model.predict(input_data)
    return prediction

if __name__ == "__main__":
    # Adjust the path to where your .pkl model file is saved
    model = load_pickle_model('models/model_name.pkl')

    # Example input data (e.g., last 5 closing prices)
    input_data = np.array([1234.56, 1236.78, 1235.89, 1237.00, 1238.12])

    # Initialize and fit the MinMaxScaler to the range of the input data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(input_data.reshape(-1, 1))

    # Preprocess the input data
    processed_input = preprocess_input(input_data, scaler)

    # Make a prediction using the loaded model
    prediction_output = predict(model, processed_input)
    print("Predicted Output:", prediction_output)
