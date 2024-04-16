from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import pickle
import re
from pathlib import Path

__version__ = "2"

BASE_DIR = Path(__file__).resolve(strict=True).parent.parent / 'models'


with open(f"{BASE_DIR}/model{__version__}.pkl", "rb") as f:
    model_ml = pickle.load(f)

def create_model(input_shape):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    checkpoint = ModelCheckpoint(model_ml, save_best_only=True)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[checkpoint])
    return history

def save_model(model):
    model.save('models/model_name.tf')
    with open('models/model_name.pkl', 'wb') as file:
        pickle.dump(model, file)




classes = [
    "Dates",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]


def predict_pipeline(text):
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', " ", text)
    text = re.sub(r"[[]]", " ", text)
    text = text.lower()
    pred = model_ml.predict([text])
    return classes[pred[0]]