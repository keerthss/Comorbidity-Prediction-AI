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

# Function to Train MLP Model and Save History
def train_mlp_model(dataset):
    print(f"Training MLP Model for {dataset}...")
    data = pd.read_csv(DATASETS[dataset])
    target = TARGET_COLUMNS[dataset]

    # Encode categorical columns
    label_encoders = {}
    for column in data.columns:
        if data[column].dtype == 'object':
            encoder = LabelEncoder()
            data[column] = encoder.fit_transform(data[column].astype(str))
            label_encoders[column] = encoder

    X = data.drop(target, axis=1)
    y = data[target]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=X.shape[1]))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32)

    model.save(f"models/{dataset}_mlp.h5")
    with open(f"models/{dataset}_mlp_history.pkl", 'wb') as f:
        pickle.dump(history.history, f)

    print(f"{dataset} MLP Model Saved Successfully ✅")

# Train MLP Models
for dataset in DATASETS.keys():
    try:
        train_mlp_model(dataset)
    except Exception as e:
        print(f"Error Training MLP Model for {dataset}: {e}")

print("All MLP Models Trained Successfully ✅")
