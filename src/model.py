from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import pickle

def create_model(input_shape):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    checkpoint = ModelCheckpoint('models/model_name.h5', save_best_only=True)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[checkpoint])
    return history

def save_model(model):
    # Save the entire model as a SavedModel.
    model.save('models/model_name.tf')
    # Save with pickle
    with open('models/model_name.pkl', 'wb') as file:
        pickle.dump(model, file)
