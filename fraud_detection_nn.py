
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Load dataset
# Replace 'credit_card.csv' with your dataset path
data = pd.read_csv("credit card.csv")

# Data preprocessing
data["type"] = data["type"].map({"PAYMENT": 1, "CASH_OUT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5})
x = data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]]
y = data["isFraud"]

# Splitting data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

# Normalizing data
scaler = StandardScaler()
x_train_norm = scaler.fit_transform(x_train)
x_test_norm = scaler.transform(x_test)

# Building the neural network
model = Sequential([
    Dense(256, activation="relu"),
    Dense(256, activation="relu"),
    Dense(256, activation="relu"),
    Dense(128, activation="relu"),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Compiling the model
model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])

# Training the model
model.fit(x_train_norm, y_train, epochs=5, batch_size=512, validation_data=(x_test_norm, y_test))

# Making predictions
y_pred = model.predict(x_test_norm)
prediction = [1 if i > 0.5 else 0 for i in y_pred.flatten()]

# Evaluating performance
accuracy = accuracy_score(y_test, prediction)
print(f"Model Accuracy: {accuracy:.2f}")

# Identifying anomalies
anomalies = []
for i in range(len(x_test_norm)):
    if y_test.reset_index(drop=True).iloc[i] != prediction[i]:
        anomalies.append(f"{i}) ACTUAL = {y_test.reset_index(drop=True).iloc[i]} PREDICTED = {prediction[i]}")

print(f"Total number of anomalies: {len(anomalies)} / {len(y_test)}")
print("Anomalies (ACTUAL vs PREDICTED):")
for anomaly in anomalies:
    print(anomaly)
