import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import History
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import pickle

# Dataset File Paths
DATASETS = {
    "heart": "./datasets/heart.csv",
    "diabetes": "./datasets/diabetes.csv",
    "parkinson": "./datasets/parkinson.csv",
    "celiac": "./datasets/celiac_disease_lab_data.csv",
    "hypertension": "./datasets/Hypertension-risk-model-main.csv",
    "kidney": "./datasets/kidney_disease.csv",
    "obesity": "./datasets/Obesity Classification.csv"
}

TARGET_COLUMNS = {
    "heart": "target",
    "diabetes": "Outcome",
    "parkinson": "status",
    "celiac": "Disease_Diagnose",
    "hypertension": "Risk",
    "kidney": "classification",
    "obesity": "Label"
}

# Preprocess Data
# Preprocess Data
def preprocess_data(dataset):
    data = pd.read_csv(DATASETS[dataset])
    target = TARGET_COLUMNS[dataset]
    X = data.drop(columns=[target])
    y = data[target]

    encoder = LabelEncoder()
    for column in X.columns:
        if X[column].dtype == 'object':
            X[column] = encoder.fit_transform(X[column].astype(str))

    # Encode Target Labels
    if y.dtype == 'object':
        y = encoder.fit_transform(y.astype(str))

    # Handle Missing Values
    X = X.fillna(X.median())

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    with open(f"models/{dataset}_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)

    return X, y

    # Handle Missing Values
    X = X.fillna(X.median())

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    with open(f"models/{dataset}_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)

    return X, y

# Train MLP Model
def train_mlp_model(dataset):
    print(f"Training MLP Model for {dataset}...")
    X, y = preprocess_data(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32)
    model.save(f"models/{dataset}_mlp.h5")
    print(f"{dataset} MLP Model Saved Successfully ✅")
    return history

# Train TabNet Model
def train_tabnet_model(dataset):
    print(f"Training TabNet Model for {dataset}...")
    X, y = preprocess_data(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = TabNetClassifier(verbose=1, optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=1e-3))
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], max_epochs=50, patience=10, batch_size=64, virtual_batch_size=16, num_workers=0, drop_last=False)
    model.save_model(f"models/{dataset}_tabnet.zip.zip")
    print(f"{dataset} TabNet Model Saved Successfully ✅")
    return model.history

# Training All Models
for dataset in DATASETS.keys():
    history_mlp = train_mlp_model(dataset)
    history_tabnet = train_tabnet_model(dataset)

print("All Models Trained Successfully ✅")
