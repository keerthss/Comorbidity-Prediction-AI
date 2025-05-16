import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
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

# MLP Model Training Function
def train_mlp_model(dataset, target_column, model_name):
    print(f"Training MLP Model for {model_name}...")
    df = pd.read_csv(DATASETS[dataset])
    if dataset == "parkinson":
        df = df.drop(columns=["name"], axis=1)
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Encode categorical features
    for column in X.columns:
        if X[column].dtype == 'object':
            encoder = LabelEncoder()
            X[column] = encoder.fit_transform(X[column].astype(str))

    if y.dtype == 'object':
        encoder = LabelEncoder()
        y = encoder.fit_transform(y.astype(str))
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = Sequential()
    model.add(Dense(128, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
    model.save(f"models/{model_name}_mlp.h5")
    print(f"{model_name} MLP Model Saved Successfully ✅")

# TabNet Model Training Function
def train_tabnet_model(dataset, target_column, model_name):
    print(f"Training TabNet Model for {model_name}...")
    df = pd.read_csv(DATASETS[dataset])
    if dataset == "parkinson":
        df = df.drop(columns=["name"], axis=1)
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Encode categorical features
    for column in X.columns:
        if X[column].dtype == 'object':
            encoder = LabelEncoder()
            X[column] = encoder.fit_transform(X[column].astype(str))

    if y.dtype == 'object':
        encoder = LabelEncoder()
        y = encoder.fit_transform(y.astype(str))
    
    X = X.fillna(0)  # Fill NaN values with 0
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = TabNetClassifier(verbose=1, optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=1e-3))
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], max_epochs=50, patience=10, batch_size=64, virtual_batch_size=16, num_workers=0, drop_last=False)
    model.save_model(f"models/{model_name}_tabnet.zip")
    print(f"{model_name} TabNet Model Saved Successfully ✅")

# Training MLP and TabNet Models for All Datasets
for dataset, target in zip(["heart", "diabetes", "parkinson", "celiac", "hypertension", "kidney", "obesity"], ["target", "Outcome", "status", "Disease_Diagnose", "Risk", "classification", "Label"]):
    train_mlp_model(dataset, target, dataset)
    train_tabnet_model(dataset, target, dataset)
