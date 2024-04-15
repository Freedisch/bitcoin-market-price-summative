import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def preprocess_data(train, test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train.drop('target', axis=1))
    y_train = train['target']
    X_test = scaler.transform(test.drop('target', axis=1))
    y_test = test['target']
    return X_train, y_train, X_test, y_test
